import type { LanguageModelV2CallOptions, LanguageModelV2StreamPart } from "@ai-sdk/provider";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { createResilienceMiddleware } from "../src/middleware";
import { attemptFallbackModel } from "../src/retries";
import type { IntermediateStateWithError, ResilienceLogger } from "../src/types";

const mockLogger: ResilienceLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

function createSuccessfulStream(chunks: LanguageModelV2StreamPart[]) {
  return {
    stream: new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        chunks.forEach(chunk => controller.enqueue(chunk));
        controller.close();
      },
    }),
    request: { body: { test: true } },
    response: { headers: { "x-test": "value" } },
  };
}

function createErrorStream(error: Error) {
  return {
    stream: new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        controller.enqueue({ type: "error" as const, error });
        controller.close();
      },
    }),
    request: {},
    response: {},
  };
}

function createPartialThenErrorStream(textChunks: LanguageModelV2StreamPart[], error: Error) {
  return {
    stream: new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        textChunks.forEach(chunk => controller.enqueue(chunk));
        controller.enqueue({ type: "error" as const, error });
        controller.close();
      },
    }),
    request: {},
    response: {},
  };
}

function createStreamResult(chunks: LanguageModelV2StreamPart[]) {
  return {
    stream: new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        chunks.forEach(chunk => controller.enqueue(chunk));
        controller.close();
      },
    }),
    request: {},
    response: {},
  };
}

type MockController = ReadableStreamDefaultController<LanguageModelV2StreamPart> & {
  enqueue: jest.Mock;
};

function createMockController(): MockController {
  return { enqueue: jest.fn(), close: jest.fn() } as unknown as MockController;
}

function createChunksState(): IntermediateStateWithError {
  return { accumulatedStreamChunks: [] };
}

const retryableError = Object.assign(new Error("Service Unavailable"), { isRetryable: true });

const baseParams: LanguageModelV2CallOptions = {
  headers: {},
  prompt: [
    { role: "system" as const, content: [{ type: "text" as const, text: "helper" }] as any },
    { role: "user" as const, content: [{ type: "text" as const, text: "Tell me more" }] as any },
  ],
};

const model = {
  modelId: "gpt-test",
  doStream: jest.fn(),
};

async function drainStream(stream: ReadableStream<LanguageModelV2StreamPart>) {
  const reader = stream.getReader();
  const chunks: LanguageModelV2StreamPart[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  return chunks;
}

describe("createResilienceMiddleware wrapStream", () => {
  const middleware = createResilienceMiddleware({
    fallbackModels: [],
    logger: mockLogger,
  });

  const middlewareWithGate = createResilienceMiddleware({
    fallbackModels: [],
    logger: mockLogger,
    shouldApply: params => params.headers?.["x-enable-resilience"] === "true",
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("skips resilience when shouldApply returns false", async () => {
    const chunks: LanguageModelV2StreamPart[] = [
      { type: "text-start", id: "0" },
      { type: "text-delta", id: "0", delta: "Hello" },
      { type: "text-end", id: "0" },
    ];
    const doStream = jest.fn(async () => createSuccessfulStream(chunks));

    const result = await middlewareWithGate.wrapStream!({
      doStream,
      params: { ...baseParams, headers: {} },
      model: model as any,
    } as any);

    expect(doStream).toHaveBeenCalledTimes(1);
    const collected = await drainStream(result.stream);
    expect(collected).toEqual(chunks);
  });

  it("wraps stream with resilience when shouldApply returns true", async () => {
    const chunks: LanguageModelV2StreamPart[] = [
      { type: "text-start", id: "0" },
      { type: "text-delta", id: "0", delta: "Resilient response" },
      { type: "text-end", id: "0" },
    ];
    const doStream = jest.fn(async () => createSuccessfulStream(chunks));

    const result = await middlewareWithGate.wrapStream!({
      doStream,
      params: {
        ...baseParams,
        headers: { "x-enable-resilience": "true" },
      },
      model: model as any,
    } as any);

    const collected = await drainStream(result.stream);
    expect(collected).toEqual(chunks);
  });

  it("always applies resilience when shouldApply is not provided", async () => {
    const chunks: LanguageModelV2StreamPart[] = [
      { type: "text-start", id: "0" },
      { type: "text-delta", id: "0", delta: "Always resilient" },
      { type: "text-end", id: "0" },
    ];
    const doStream = jest.fn(async () => createSuccessfulStream(chunks));

    const result = await middleware.wrapStream!({
      doStream,
      params: baseParams,
      model: model as any,
    } as any);

    const collected = await drainStream(result.stream);
    expect(collected).toEqual(chunks);
  });

  it("propagates request and response metadata on successful stream", async () => {
    const chunks: LanguageModelV2StreamPart[] = [{ type: "text-start", id: "0" }];
    const doStream = jest.fn(async () => createSuccessfulStream(chunks));

    const result = await middleware.wrapStream!({
      doStream,
      params: baseParams,
      model: model as any,
    } as any);

    await drainStream(result.stream);

    expect(await (result.request as Promise<{ body?: unknown } | undefined>)).toEqual({ body: { test: true } });
    expect(await (result.response as Promise<{ headers?: Record<string, string> } | undefined>)).toEqual({
      headers: { "x-test": "value" },
    });
  });

  it("emits error chunk when stream fails with non-retryable error", async () => {
    const errorChunk: LanguageModelV2StreamPart = {
      type: "error",
      error: new Error("Non-retryable failure"),
    };
    const doStream = jest.fn(async () => ({
      stream: new ReadableStream<LanguageModelV2StreamPart>({
        start(controller) {
          controller.enqueue({ type: "text-start", id: "0" });
          controller.enqueue(errorChunk);
          controller.close();
        },
      }),
      request: {},
      response: {},
    }));

    const result = await middleware.wrapStream!({
      doStream,
      params: baseParams,
      model: model as any,
    } as any);

    const collected = await drainStream(result.stream);
    expect(collected.length).toBeGreaterThanOrEqual(1);
    expect(collected[collected.length - 1].type).toBe("error");
  });
});

describe("fallback chain ordering", () => {
  const mockHaikuDoStream = jest.fn();
  const mockGeminiDoStream = jest.fn();

  const haikuModel = { doStream: mockHaikuDoStream } as any;
  const geminiModel = { doStream: mockGeminiDoStream } as any;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  async function runFallbackChain(opts?: { chunksState?: IntermediateStateWithError }) {
    const controller = createMockController();
    const chunksState = opts?.chunksState ?? createChunksState();
    chunksState.errorDuringStreamAttempt ??= retryableError;

    const haikuResult = await attemptFallbackModel({
      controller,
      fallbackModel: haikuModel,
      fallbackModelId: "claude-haiku-4-5-20251001",
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
    });

    let geminiResult: Awaited<ReturnType<typeof attemptFallbackModel>> = null;
    if (!haikuResult) {
      geminiResult = await attemptFallbackModel({
        controller,
        fallbackModel: geminiModel,
        fallbackModelId: "gemini-2.5-flash",
        logger: mockLogger,
        originalModelId: "gpt-test",
        params: baseParams,
        chunksState,
        maxSameModelRetries: 1,
      });
    }

    return { haikuResult, geminiResult, controller, chunksState };
  }

  it("tries haiku first and stops if haiku succeeds", async () => {
    mockHaikuDoStream.mockImplementation(async () =>
      createStreamResult([
        { type: "text-start", id: "0" },
        { type: "text-delta", id: "0", delta: "haiku response" },
        { type: "text-end", id: "0" },
      ]),
    );

    const { haikuResult, geminiResult } = await runFallbackChain();

    expect(haikuResult).toBeTruthy();
    expect(geminiResult).toBeNull();
    expect(mockHaikuDoStream).toHaveBeenCalledTimes(1);
    expect(mockGeminiDoStream).not.toHaveBeenCalled();
  });

  it("falls through to gemini when haiku fails with retryable error", async () => {
    mockHaikuDoStream.mockImplementation(async () => createErrorStream(retryableError));
    mockGeminiDoStream.mockImplementation(async () =>
      createStreamResult([
        { type: "text-start", id: "0" },
        { type: "text-delta", id: "0", delta: "gemini response" },
        { type: "text-end", id: "0" },
      ]),
    );

    const { haikuResult, geminiResult } = await runFallbackChain();

    expect(haikuResult).toBeNull();
    expect(geminiResult).toBeTruthy();
    expect(mockHaikuDoStream).toHaveBeenCalledTimes(1);
    expect(mockGeminiDoStream).toHaveBeenCalledTimes(1);
  });

  it("returns null for both when both fail", async () => {
    mockHaikuDoStream.mockImplementation(async () => createErrorStream(retryableError));
    mockGeminiDoStream.mockImplementation(async () => createErrorStream(retryableError));

    const { haikuResult, geminiResult } = await runFallbackChain();

    expect(haikuResult).toBeNull();
    expect(geminiResult).toBeNull();
    expect(mockHaikuDoStream).toHaveBeenCalledTimes(1);
    expect(mockGeminiDoStream).toHaveBeenCalledTimes(1);
  });

  it("accumulates partial chunks from haiku and passes them to gemini", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    chunksState.errorDuringStreamAttempt = retryableError;

    mockHaikuDoStream.mockImplementation(async () =>
      createPartialThenErrorStream(
        [
          { type: "text-start", id: "0" },
          { type: "text-delta", id: "0", delta: "partial from haiku" },
        ],
        retryableError,
      ),
    );

    const haikuResult = await attemptFallbackModel({
      controller,
      fallbackModel: haikuModel,
      fallbackModelId: "claude-haiku-4-5-20251001",
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
    });

    expect(haikuResult).toBeNull();
    expect(chunksState.accumulatedStreamChunks.length).toBeGreaterThan(0);

    mockGeminiDoStream.mockImplementation(async () =>
      createStreamResult([
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: "completed by gemini" },
        { type: "text-end", id: "1" },
      ]),
    );

    const geminiResult = await attemptFallbackModel({
      controller,
      fallbackModel: geminiModel,
      fallbackModelId: "gemini-2.5-flash",
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
    });

    expect(geminiResult).toBeTruthy();
    const enqueuedChunks = (controller.enqueue as jest.Mock).mock.calls.map(c => c[0] as LanguageModelV2StreamPart);
    const textDeltas = enqueuedChunks.filter(c => c.type === "text-delta");
    expect(textDeltas).toHaveLength(2);
    expect((textDeltas[0] as any).delta).toBe("partial from haiku");
    expect((textDeltas[1] as any).delta).toBe("completed by gemini");
  });
});
