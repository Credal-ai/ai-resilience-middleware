import type { LanguageModelV2CallOptions, LanguageModelV2StreamPart, SharedV2Headers } from "@ai-sdk/provider";
import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { APICallError, DownloadError } from "ai";
import { ReadableStream, type ReadableStreamDefaultController } from "node:stream/web";
import { attemptFallbackModel, attemptSameModel } from "../src/retries";
import type { IntermediateStateWithError, ResilienceLogger } from "../src/types";

const mockFallbackModelDoStream = jest.fn();
const mockFallbackModel = {
  doStream: mockFallbackModelDoStream,
};

type MockController = ReadableStreamDefaultController<LanguageModelV2StreamPart> & {
  enqueue: jest.Mock;
  close: jest.Mock;
};

function createMockController(): MockController {
  return {
    enqueue: jest.fn(),
    close: jest.fn(),
  } as unknown as MockController;
}

function createStream(chunks: LanguageModelV2StreamPart[]): ReadableStream<LanguageModelV2StreamPart> {
  return new ReadableStream({
    start(controller) {
      chunks.forEach(chunk => controller.enqueue(chunk));
      controller.close();
    },
  });
}

function createStreamResult(chunks: LanguageModelV2StreamPart[]): {
  stream: ReadableStream<LanguageModelV2StreamPart>;
  request?: { body?: unknown };
  response?: { headers?: SharedV2Headers };
} {
  return {
    stream: createStream(chunks),
    request: {},
    response: {},
  };
}

function createStreamWithError(chunksBeforeError: LanguageModelV2StreamPart[], error?: Error) {
  const errorChunk: LanguageModelV2StreamPart = {
    type: "error",
    error: error || new Error("mid-stream failure"),
  };
  return createStreamResult([...chunksBeforeError, errorChunk]);
}

function createChunksState(): IntermediateStateWithError {
  return { accumulatedStreamChunks: [] };
}

const baseParams: LanguageModelV2CallOptions = {
  prompt: [
    {
      role: "system" as const,
      content: [{ type: "text" as const, text: "You are helpful" }] as any,
    },
    {
      role: "user" as const,
      content: [{ type: "text" as const, text: "Tell me something" }] as any,
    },
  ],
};

const mockLogger: ResilienceLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

describe("attemptSameModel", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("returns immediately when the first attempt succeeds", async () => {
    const controller = createMockController();
    const successfulStream = createStreamResult([{ type: "text-start", id: "0" }]);
    const doStream = jest.fn(async () => successfulStream as any);
    const result = await attemptSameModel({
      controller,
      doStream,
      modelDoStream: jest.fn() as any,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState: createChunksState(),
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toEqual({ request: successfulStream.request, response: successfulStream.response });
    expect(doStream).toHaveBeenCalledTimes(1);
  });

  it("retries after a mid-stream error chunk and succeeds", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const retryableError = Object.assign(new Error("mid-stream failure"), { isRetryable: true });
    const firstAttempt = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], retryableError) as any,
    );
    const retryAttempt = jest.fn(async () => createStreamResult([{ type: "text-start", id: "1" }]) as any);

    const attemptPromise = attemptSameModel({
      controller,
      doStream: firstAttempt,
      modelDoStream: retryAttempt,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    await jest.runAllTimersAsync();
    const result = await attemptPromise;

    expect(result).toBeTruthy();
    expect(firstAttempt).toHaveBeenCalledTimes(1);
    expect(retryAttempt).toHaveBeenCalledTimes(1);
    expect(chunksState.accumulatedStreamChunks.length).toBe(0);
  });

  it("returns null when retries are exhausted", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const failingStream = jest.fn(
      async () =>
        createStreamWithError([
          { type: "text-start", id: "0" },
          { type: "text-delta", id: "0", delta: "partial" },
        ]) as any,
    );

    const result = await attemptSameModel({
      controller,
      doStream: failingStream,
      modelDoStream: failingStream,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 0,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toBeNull();
    expect(chunksState.accumulatedStreamChunks.length).toBeGreaterThan(0);
  });

  it("fails immediately for non-retryable APICallError (400 Bad Request)", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const nonRetryableError = new APICallError({
      message: "Bad Request",
      url: "https://api.example.com",
      requestBodyValues: {},
      statusCode: 400,
      isRetryable: false,
    });
    const failingStream = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], nonRetryableError) as any,
    );

    const result = await attemptSameModel({
      controller,
      doStream: failingStream,
      modelDoStream: failingStream,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toBeNull();
    expect(failingStream).toHaveBeenCalledTimes(1);
    expect(chunksState.accumulatedStreamChunks.length).toBeGreaterThan(0);
  });

  it("retries for retryable APICallError (503 Service Unavailable)", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const retryableError = new APICallError({
      message: "Service Unavailable",
      url: "https://api.example.com",
      requestBodyValues: {},
      statusCode: 503,
      isRetryable: true,
    });
    const firstAttempt = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], retryableError) as any,
    );
    const retryAttempt = jest.fn(async () => createStreamResult([{ type: "text-start", id: "1" }]) as any);

    const attemptPromise = attemptSameModel({
      controller,
      doStream: firstAttempt,
      modelDoStream: retryAttempt,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    await jest.runAllTimersAsync();
    const result = await attemptPromise;

    expect(result).toBeTruthy();
    expect(firstAttempt).toHaveBeenCalledTimes(1);
    expect(retryAttempt).toHaveBeenCalledTimes(1);
    expect(chunksState.accumulatedStreamChunks.length).toBe(0);
  });

  it("retries for DownloadError with retryable status code (429)", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const retryableError = new DownloadError({
      url: "https://api.example.com",
      statusCode: 429,
      message: "Too Many Requests",
    });
    const firstAttempt = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], retryableError) as any,
    );
    const retryAttempt = jest.fn(async () => createStreamResult([{ type: "text-start", id: "1" }]) as any);

    const attemptPromise = attemptSameModel({
      controller,
      doStream: firstAttempt,
      modelDoStream: retryAttempt,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    await jest.runAllTimersAsync();
    const result = await attemptPromise;

    expect(result).toBeTruthy();
    expect(firstAttempt).toHaveBeenCalledTimes(1);
    expect(retryAttempt).toHaveBeenCalledTimes(1);
  });

  it("fails immediately for DownloadError with non-retryable status code (404)", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const nonRetryableError = new DownloadError({
      url: "https://api.example.com",
      statusCode: 404,
      message: "Not Found",
    });
    const failingStream = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], nonRetryableError) as any,
    );

    const result = await attemptSameModel({
      controller,
      doStream: failingStream,
      modelDoStream: failingStream,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toBeNull();
    expect(failingStream).toHaveBeenCalledTimes(1);
  });

  it("retries for transient network error (timeout)", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const networkError = new Error("Request timeout");
    const firstAttempt = jest.fn(
      async () => createStreamWithError([{ type: "text-start", id: "0" }], networkError) as any,
    );
    const retryAttempt = jest.fn(async () => createStreamResult([{ type: "text-start", id: "1" }]) as any);

    const attemptPromise = attemptSameModel({
      controller,
      doStream: firstAttempt,
      modelDoStream: retryAttempt,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    await jest.runAllTimersAsync();
    const result = await attemptPromise;

    expect(result).toBeTruthy();
    expect(firstAttempt).toHaveBeenCalledTimes(1);
    expect(retryAttempt).toHaveBeenCalledTimes(1);
  });

  it("captures failure immediately after a tool-result chunk", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const toolCallChunk: LanguageModelV2StreamPart = {
      type: "tool-call",
      toolCallId: "tool-1",
      toolName: "demoTool",
      input: "{}",
      providerExecuted: false,
    };
    const toolResultChunk: LanguageModelV2StreamPart = {
      type: "tool-result",
      toolCallId: "tool-1",
      toolName: "demoTool",
      result: { summary: "demo" },
      providerExecuted: false,
    };
    const toolResultError = Object.assign(new Error("tool result failure"), { isRetryable: true });
    const failingStream = jest.fn(
      async () => createStreamWithError([toolCallChunk, toolResultChunk], toolResultError) as any,
    );

    const result = await attemptSameModel({
      controller,
      doStream: failingStream,
      modelDoStream: failingStream,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 0,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toBeNull();
    expect(controller.enqueue).toHaveBeenCalledWith(toolCallChunk);
    expect(controller.enqueue).toHaveBeenCalledWith(toolResultChunk);
    expect(chunksState.errorDuringStreamAttempt).toBe(toolResultError);
  });

  it("handles a reasoning delta failure when the error is marked retryable", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const reasoningStart: LanguageModelV2StreamPart = {
      type: "reasoning-start",
      id: "reasoning-1",
    };
    const reasoningDelta: LanguageModelV2StreamPart = {
      type: "reasoning-delta",
      id: "reasoning-1",
      delta: "Thinking step...",
    };
    const reasoningError = Object.assign(new Error("reasoning failure"), { isRetryable: true });
    const failingStream = jest.fn(
      async () => createStreamWithError([reasoningStart, reasoningDelta], reasoningError) as any,
    );

    const result = await attemptSameModel({
      controller,
      doStream: failingStream,
      modelDoStream: failingStream,
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 0,
      initialRetryDelayMs: 1000,
      maxRetryDelayMs: 8000,
    });

    expect(result).toBeNull();
    expect(controller.enqueue).toHaveBeenCalledWith(reasoningStart);
    expect(controller.enqueue).toHaveBeenCalledWith(reasoningDelta);
    expect(chunksState.errorDuringStreamAttempt).toBe(reasoningError);
    expect((chunksState.errorDuringStreamAttempt as any)?.isRetryable).toBe(true);
  });
});

describe("attemptFallbackModel", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("skips cross-provider fallback for non-retryable error", async () => {
    const controller = createMockController();
    const chunksState = createChunksState();
    const nonRetryableError = new APICallError({
      message: "Bad Request",
      url: "https://api.example.com",
      requestBodyValues: {},
      statusCode: 400,
      isRetryable: false,
    });
    chunksState.errorDuringStreamAttempt = nonRetryableError;

    const result = await attemptFallbackModel({
      controller,
      fallbackModel: mockFallbackModel as any,
      fallbackModelId: "fallback-model",
      logger: mockLogger,
      originalModelId: "gpt-test",
      params: baseParams,
      chunksState,
      maxSameModelRetries: 1,
    });

    expect(result).toBeNull();
    expect(mockFallbackModelDoStream).not.toHaveBeenCalled();
  });
});
