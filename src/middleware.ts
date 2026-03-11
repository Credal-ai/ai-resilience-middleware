/* eslint-disable @typescript-eslint/consistent-type-assertions */
import type {
  LanguageModelV2CallOptions,
  LanguageModelV2Middleware,
  LanguageModelV2StreamPart,
  SharedV2Headers,
} from "@ai-sdk/provider";
import { createDefaultLogger } from "./logger";
import { attemptFallbackModel, attemptSameModel } from "./retries";
import type { IntermediateStateWithError, ResilienceMiddlewareConfig } from "./types";
import { isRetryableError, shouldAttemptCrossProviderFallback } from "./utils";

const DEFAULT_MAX_SAME_MODEL_RETRIES = 1;
const DEFAULT_INITIAL_RETRY_DELAY_MS = 1000;
const DEFAULT_MAX_RETRY_DELAY_MS = 8000;

export function createResilienceMiddleware(config: ResilienceMiddlewareConfig): LanguageModelV2Middleware {
  const logger = config.logger ?? createDefaultLogger();
  const shouldApply = config.shouldApply ?? (() => true);
  const isFallbackEnabled = config.isFallbackEnabled ?? (() => true);
  const maxSameModelRetries = config.maxSameModelRetries ?? DEFAULT_MAX_SAME_MODEL_RETRIES;
  const initialRetryDelayMs = config.initialRetryDelayMs ?? DEFAULT_INITIAL_RETRY_DELAY_MS;
  const maxRetryDelayMs = config.maxRetryDelayMs ?? DEFAULT_MAX_RETRY_DELAY_MS;

  return {
    wrapGenerate: async ({ doGenerate }) => {
      return await doGenerate();
    },
    wrapStream: async ({ doStream, params, model }) => {
      if (!shouldApply(params)) {
        return await doStream();
      }

      const originalModelId = model.modelId;
      const chunksStateWithError: IntermediateStateWithError = {
        accumulatedStreamChunks: [],
      };

      let resolveRequest: (val: { body?: unknown } | undefined) => void;
      let resolveResponse: (val: { headers?: SharedV2Headers } | undefined) => void;
      const requestPromise = new Promise<{ body?: unknown } | undefined>(resolve => {
        resolveRequest = resolve;
      });
      const responsePromise = new Promise<{ headers?: SharedV2Headers } | undefined>(resolve => {
        resolveResponse = resolve;
      });

      const doStreamResult = {
        stream: new ReadableStream<LanguageModelV2StreamPart>({
          start(controller) {
            const completeWithResult = (attemptResult: {
              request?: { body?: unknown };
              response?: { headers?: SharedV2Headers };
            }) => {
              resolveRequest(attemptResult.request);
              resolveResponse(attemptResult.response);
              controller.close();
            };

            const runStreamAttempt = async () => {
              const sameModelResult = await attemptSameModel({
                controller,
                doStream,
                modelDoStream: retryParams => model.doStream(retryParams),
                logger,
                originalModelId,
                params,
                chunksState: chunksStateWithError,
                maxSameModelRetries,
                initialRetryDelayMs,
                maxRetryDelayMs,
                onAttemptFailed: config.onAttemptFailed,
              });

              if (sameModelResult) {
                completeWithResult(sameModelResult);
                return;
              }

              const previousError = chunksStateWithError.errorDuringStreamAttempt;
              const shouldFallback = shouldAttemptCrossProviderFallback(previousError);

              if (shouldFallback) {
                for (const fallbackConfig of config.fallbackModels) {
                  if (!isFallbackEnabled(params, fallbackConfig.modelId)) {
                    continue;
                  }

                  const fallbackModel = fallbackConfig.createModel();
                  const fallbackResult = await attemptFallbackModel({
                    controller,
                    fallbackModel,
                    fallbackModelId: fallbackConfig.modelId,
                    logger,
                    originalModelId,
                    params,
                    chunksState: chunksStateWithError,
                    maxSameModelRetries,
                    onAttemptFailed: config.onAttemptFailed,
                  });

                  if (fallbackResult) {
                    completeWithResult(fallbackResult);
                    return;
                  }

                  const fallbackError = chunksStateWithError.errorDuringStreamAttempt;
                  if (fallbackError && !isRetryableError(fallbackError)) {
                    break;
                  }
                }
              }

              const finalError =
                chunksStateWithError.errorDuringStreamAttempt ||
                new Error("ResilienceMiddleware Error: retry loop completed without any result");
              throw finalError;
            };

            runStreamAttempt().catch(error => {
              logger.error("ResilienceMiddleware: resilience logic depleted, throwing error", {
                model_id: originalModelId,
                error,
              });
              controller.enqueue({
                type: "error",
                error: error,
              });
              controller.close();
            });
          },
        }),
        request: requestPromise,
        response: responsePromise,
      } as {
        stream: ReadableStream<LanguageModelV2StreamPart>;
        request?: { body?: unknown };
        response?: { headers?: SharedV2Headers };
      };

      return doStreamResult;
    },
  };
}

export function shouldSkipResilienceLogic(
  params: LanguageModelV2CallOptions,
  shouldApply: (params: LanguageModelV2CallOptions) => boolean,
): boolean {
  return !shouldApply(params);
}
