import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
  SharedV2Headers,
} from "@ai-sdk/provider";
import { serializeError } from "serialize-error";
import type { IntermediateStateWithError, ResilienceAttemptPhase, ResilienceLogger } from "./types";
import {
  extractCompletionPartialText,
  getRetryDelay,
  isRetryableError,
  normalizeStreamError,
  reconstructPromptWithPartialChunks,
  shouldAttemptCrossProviderFallback,
} from "./utils";

type DoStreamFn = () => ReturnType<LanguageModelV2["doStream"]>;
type DoStreamResult = Awaited<ReturnType<DoStreamFn>>;
type AttemptResult = {
  request?: { body?: unknown };
  response?: { headers?: SharedV2Headers };
};

type AttemptSameModelOptions = {
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>;
  doStream: DoStreamFn;
  modelDoStream: (params: LanguageModelV2CallOptions) => PromiseLike<DoStreamResult>;
  logger: ResilienceLogger;
  originalModelId: string;
  params: LanguageModelV2CallOptions;
  chunksState: IntermediateStateWithError;
  maxSameModelRetries: number;
  initialRetryDelayMs: number;
  maxRetryDelayMs: number;
  onAttemptFailed?: (details: {
    phase: ResilienceAttemptPhase;
    attemptNumber: number;
    modelId: string;
    error: Error;
    isRetryable: boolean;
    partialResponse: string;
    chunksAccumulated: number;
    durationMs: number;
    timestamp: string;
    rawRequest?: unknown;
  }) => void | Promise<void>;
};

export async function attemptSameModel(options: AttemptSameModelOptions): Promise<AttemptResult | null> {
  const {
    controller,
    logger,
    doStream,
    modelDoStream,
    originalModelId,
    params,
    chunksState,
    maxSameModelRetries,
    initialRetryDelayMs,
    maxRetryDelayMs,
    onAttemptFailed,
  } = options;

  for (let attemptNumber = 0; attemptNumber <= maxSameModelRetries; attemptNumber++) {
    const attemptStartTime = Date.now();
    const isRetry = attemptNumber > 0;
    const modelParams = isRetry
      ? { ...params, prompt: reconstructPromptWithPartialChunks(params.prompt, chunksState.accumulatedStreamChunks) }
      : params;
    const effectiveDoStream = isRetry ? async () => await modelDoStream(modelParams) : doStream;

    if (isRetry) {
      const retryDelay = getRetryDelay(attemptNumber, initialRetryDelayMs, maxRetryDelayMs);
      logger.warn("ResilienceMiddleware: Retrying with same model", {
        attempt: attemptNumber + 1,
        total_attempts: maxSameModelRetries + 1,
        delay_ms: retryDelay,
        model_id: originalModelId,
        chunks_before_failure: chunksState.accumulatedStreamChunks.length,
      });
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }

    try {
      const attemptResult = await streamAttempt({
        controller,
        doStreamFn: effectiveDoStream,
        attemptNumber,
        modelId: originalModelId,
        phase: "same_model_retry",
        onAttemptFailed: chunks => {
          accumulateChunks(chunksState, chunks);
        },
        onAttemptSucceeded: () => {
          resetAccumulatedChunks(chunksState);
        },
        logger,
      });
      chunksState.errorDuringStreamAttempt = undefined;
      return attemptResult;
    } catch (error) {
      const durationMs = Date.now() - attemptStartTime;
      const normalizedError = error as Error;
      chunksState.errorDuringStreamAttempt = normalizedError;
      const isRetryable = isRetryableError(normalizedError);

      logger.error("ResilienceMiddleware: Stream attempt failed", {
        attempt: attemptNumber + 1,
        model_id: originalModelId,
        is_retryable: isRetryable,
        will_retry: isRetryable && attemptNumber < maxSameModelRetries,
        error: serializeError(stripRequestFromError(normalizedError)),
      });

      if (onAttemptFailed) {
        Promise.resolve(
          onAttemptFailed({
            phase: "same_model_retry",
            attemptNumber: attemptNumber + 1,
            modelId: originalModelId,
            error: normalizedError,
            isRetryable,
            partialResponse: extractCompletionPartialText(chunksState.accumulatedStreamChunks),
            chunksAccumulated: chunksState.accumulatedStreamChunks.length,
            durationMs,
            timestamp: new Date().toISOString(),
            rawRequest: modelParams,
          }),
        ).catch(callbackError => {
          logger.error("ResilienceMiddleware: onAttemptFailed callback error", {
            model_id: originalModelId,
            attempt: attemptNumber + 1,
            callback_error: serializeError(callbackError),
          });
        });
      }

      if (!isRetryable || attemptNumber >= maxSameModelRetries) {
        break;
      }
    }
  }

  logger.warn("ResilienceMiddleware: Same model retries exhausted", {
    model_id: originalModelId,
    total_attempts: maxSameModelRetries + 1,
    error_message: chunksState.errorDuringStreamAttempt?.message,
  });
  return null;
}

type AttemptFallbackModelOptions = {
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>;
  fallbackModel: LanguageModelV2;
  fallbackModelId: string;
  logger: ResilienceLogger;
  originalModelId: string;
  params: LanguageModelV2CallOptions;
  chunksState: IntermediateStateWithError;
  maxSameModelRetries: number;
  onAttemptFailed?: AttemptSameModelOptions["onAttemptFailed"];
};

export async function attemptFallbackModel(options: AttemptFallbackModelOptions): Promise<AttemptResult | null> {
  const {
    controller,
    fallbackModel,
    fallbackModelId,
    logger,
    originalModelId,
    params,
    chunksState,
    maxSameModelRetries,
    onAttemptFailed,
  } = options;

  const error = chunksState.errorDuringStreamAttempt;
  if (error && !shouldAttemptCrossProviderFallback(error)) {
    logger.error("ResilienceMiddleware: Skipping cross-provider fallback for non-fallbackable error", {
      original_model: originalModelId,
      error: serializeError(error),
    });
    return null;
  }

  logger.warn("ResilienceMiddleware: Switching to cross-provider fallback", {
    original_model: originalModelId,
    fallback_model: fallbackModelId,
    chunks_accumulated: chunksState.accumulatedStreamChunks.length,
  });

  const reconstructedParams = {
    ...params,
    prompt: reconstructPromptWithPartialChunks(params.prompt, chunksState.accumulatedStreamChunks),
  };

  const attemptStartTime = Date.now();
  const attemptNumber = maxSameModelRetries + 1;

  try {
    const attemptResult = await streamAttempt({
      controller,
      doStreamFn: async () => await fallbackModel.doStream(reconstructedParams),
      attemptNumber,
      modelId: fallbackModelId,
      phase: "cross_provider_fallback",
      onAttemptFailed: chunks => {
        accumulateChunks(chunksState, chunks);
      },
      onAttemptSucceeded: () => {
        resetAccumulatedChunks(chunksState);
      },
      logger,
    });

    chunksState.errorDuringStreamAttempt = undefined;
    return attemptResult;
  } catch (fallbackError) {
    const durationMs = Date.now() - attemptStartTime;
    const normalizedError = fallbackError as Error;
    chunksState.errorDuringStreamAttempt = normalizedError;
    const isRetryable = isRetryableError(normalizedError);

    logger.error("ResilienceMiddleware: Cross-provider fallback failed", {
      original_model: originalModelId,
      fallback_model: fallbackModelId,
      is_retryable: isRetryable,
      error: serializeError(normalizedError),
    });

    if (onAttemptFailed) {
      Promise.resolve(
        onAttemptFailed({
          phase: "cross_provider_fallback",
          attemptNumber: attemptNumber + 1,
          modelId: fallbackModelId,
          error: normalizedError,
          isRetryable,
          partialResponse: extractCompletionPartialText(chunksState.accumulatedStreamChunks),
          chunksAccumulated: chunksState.accumulatedStreamChunks.length,
          durationMs,
          timestamp: new Date().toISOString(),
          rawRequest: reconstructedParams,
        }),
      ).catch(callbackError => {
        logger.error("ResilienceMiddleware: onAttemptFailed callback error", {
          original_model: originalModelId,
          fallback_model: fallbackModelId,
          callback_error: serializeError(callbackError),
        });
      });
    }

    return null;
  }
}

type StreamAttemptOptions = {
  controller: ReadableStreamDefaultController<LanguageModelV2StreamPart>;
  doStreamFn: () => PromiseLike<{
    stream: ReadableStream<LanguageModelV2StreamPart>;
    request?: { body?: unknown };
    response?: { headers?: Record<string, string> };
  }>;
  attemptNumber: number;
  modelId: string;
  phase: ResilienceAttemptPhase;
  onAttemptFailed: (chunks: LanguageModelV2StreamPart[]) => void;
  onAttemptSucceeded: (chunks: LanguageModelV2StreamPart[]) => void;
  logger: ResilienceLogger;
};

async function streamAttempt({
  controller,
  doStreamFn,
  attemptNumber,
  modelId,
  phase,
  onAttemptFailed,
  onAttemptSucceeded,
  logger,
}: StreamAttemptOptions): Promise<{
  request?: { body?: unknown };
  response?: { headers?: Record<string, string> };
}> {
  const accumulatedChunks: LanguageModelV2StreamPart[] = [];
  let chunkCount = 0;

  const streamResult = await doStreamFn();
  const streamReader = streamResult.stream.getReader();

  try {
    while (true) {
      const { done, value } = await streamReader.read();
      if (value) {
        chunkCount++;
        accumulatedChunks.push(value);
        const chunkToProcess: LanguageModelV2StreamPart = value;

        if (chunkToProcess.type === "error") {
          const normalizedErrorFromChunk = normalizeStreamError(chunkToProcess.error);
          const isRetryable = isRetryableError(normalizedErrorFromChunk);
          logger.warn("ResilienceMiddleware: Error chunk detected mid-stream", {
            attempt: attemptNumber + 1,
            phase,
            model_id: modelId,
            is_retryable: isRetryable,
            error: serializeError(normalizedErrorFromChunk),
          });
          throw normalizedErrorFromChunk;
        }

        controller.enqueue(chunkToProcess);
      }

      if (done) break;
    }

    streamReader.releaseLock();
    onAttemptSucceeded(accumulatedChunks);

    logger.info("ResilienceMiddleware: Stream attempt succeeded", {
      attempt: attemptNumber + 1,
      phase,
      model_id: modelId,
      chunks: chunkCount,
    });

    return {
      request: streamResult.request,
      response: streamResult.response,
    };
  } catch (error) {
    streamReader.releaseLock();
    onAttemptFailed(accumulatedChunks);
    throw error;
  }
}

function accumulateChunks(state: IntermediateStateWithError, chunks: LanguageModelV2StreamPart[]) {
  state.accumulatedStreamChunks = [...state.accumulatedStreamChunks, ...chunks];
}

function resetAccumulatedChunks(state: IntermediateStateWithError) {
  state.accumulatedStreamChunks = [];
}

function stripRequestFromError(error: any) {
  if (error == null || typeof error !== "object") {
    return error;
  }
  const { requestBodyValues: _, ...rest } = error;
  return rest;
}
