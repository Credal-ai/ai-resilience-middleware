import type { LanguageModelV2Prompt, LanguageModelV2StreamPart } from "@ai-sdk/provider";
import { describe, expect, it } from "@jest/globals";
import { APICallError, DownloadError } from "ai";
import {
  getRetryDelay,
  isRetryableError,
  normalizeStreamError,
  reconstructPromptWithPartialChunks,
} from "../src/utils";

const toPrompt = (messages: unknown[]): LanguageModelV2Prompt => messages as LanguageModelV2Prompt;

describe("reconstructPromptWithPartialChunks", () => {
  it("consolidates text and reasoning segments while filtering empty assistants", () => {
    const prompt = toPrompt([
      { role: "system", content: "Chef agent" },
      { role: "assistant", content: [{ type: "text", text: "   " }] },
      { role: "user", content: [{ type: "text", text: "Tell me about apple pie" }] },
    ]);
    const partialChunks: LanguageModelV2StreamPart[] = [
      { type: "text-start", id: "text-1" },
      { type: "text-delta", id: "text-1", delta: "Here is the" },
      { type: "text-delta", id: "text-1", delta: " summary." },
      { type: "text-end", id: "text-1" },
      { type: "reasoning-start", id: "reason-1" },
      { type: "reasoning-delta", id: "reason-1", delta: "Chain-of-thought" },
    ];

    const result = reconstructPromptWithPartialChunks(prompt, partialChunks);

    expect(result).toHaveLength(3);
    const assistant = result[2];
    expect(assistant.role).toBe("assistant");
    expect(assistant.content).toEqual([
      { type: "text", text: "Here is the summary." },
      { type: "reasoning", text: "Chain-of-thought" },
    ]);
  });

  it("replays provider-executed tool calls/results as assistant + tool messages", () => {
    const prompt = toPrompt([
      { role: "system", content: "Chef agent" },
      { role: "user", content: [{ type: "text", text: "Tell me about apple pie" }] },
    ]);
    const toolCallChunk: LanguageModelV2StreamPart = {
      type: "tool-call",
      toolCallId: "call-1",
      toolName: "recipeAction",
      input: '{"query":"apple pie"}',
      providerExecuted: true,
      providerMetadata: { openai: { itemId: "fc_1" } },
    };
    const toolResultChunk: LanguageModelV2StreamPart = {
      type: "tool-result",
      toolCallId: "call-1",
      toolName: "recipeAction",
      result: { summary: "SECRET ingredient" },
      providerExecuted: true,
      providerMetadata: { openai: { itemId: "fc_1" } },
    };

    const result = reconstructPromptWithPartialChunks(prompt, [toolCallChunk, toolResultChunk]);

    expect(result).toHaveLength(4);
    const assistantMessage = result[2];
    const toolMessage = result[3];

    expect(assistantMessage.role).toBe("assistant");
    expect(assistantMessage.content).toEqual([
      {
        type: "tool-call",
        toolCallId: "call-1",
        toolName: "recipeAction",
        input: { query: "apple pie" },
        providerExecuted: true,
        providerOptions: { openai: { itemId: "fc_1" } },
      },
    ]);

    expect(toolMessage.role).toBe("tool");
    expect(toolMessage.content).toEqual([
      {
        type: "tool-result",
        toolCallId: "call-1",
        toolName: "recipeAction",
        output: { type: "json", value: { summary: "SECRET ingredient" } },
        providerOptions: { openai: { itemId: "fc_1" } },
      },
    ]);
  });

  it("drops orphaned provider-executed tool calls without results", () => {
    const prompt = toPrompt([
      { role: "system", content: "Chef agent" },
      { role: "user", content: [{ type: "text", text: "Tell me about apple pie" }] },
    ]);
    const orphanedToolCall: LanguageModelV2StreamPart = {
      type: "tool-call",
      toolCallId: "call-2",
      toolName: "recipeAction",
      input: '{"query":"orphan"}',
      providerExecuted: true,
      providerMetadata: { openai: { itemId: "fc_2" } },
    };

    const result = reconstructPromptWithPartialChunks(prompt, [orphanedToolCall]);

    expect(result).toEqual(prompt);
  });
});

describe("normalizeStreamError", () => {
  it("returns Error instances for primitive inputs", () => {
    expect(normalizeStreamError("boom").message).toBe("boom");
    expect(normalizeStreamError({ code: 42 }).message).toBe('{"code":42}');
  });
});

describe("getRetryDelay", () => {
  it("applies exponential backoff capped at 8 seconds", () => {
    expect(getRetryDelay(1)).toBe(1000);
    expect(getRetryDelay(2)).toBe(2000);
    expect(getRetryDelay(3)).toBe(4000);
    expect(getRetryDelay(5)).toBe(8000);
  });
});

describe("isRetryableError", () => {
  it("identifies retryable AI SDK errors", () => {
    const retryableApiError = new APICallError({
      message: "bad gateway",
      url: "https://api",
      requestBodyValues: {},
      statusCode: 502,
      isRetryable: true,
    });
    const retryableDownloadError = new DownloadError({
      url: "https://api",
      statusCode: 429,
      message: "rate limit",
    });
    const taggedError = Object.assign(new Error("retry please"), { isRetryable: true });
    const networkError = new Error("Network connection lost");

    expect(isRetryableError(retryableApiError)).toBe(true);
    expect(isRetryableError(retryableDownloadError)).toBe(true);
    expect(isRetryableError(taggedError)).toBe(true);
    expect(isRetryableError(networkError)).toBe(true);
  });

  it("treats non-retryable cases as false", () => {
    const nonRetryable = new APICallError({
      message: "Bad Request",
      url: "https://api",
      requestBodyValues: {},
      statusCode: 400,
      isRetryable: false,
    });
    const download404 = new DownloadError({
      url: "https://api",
      statusCode: 404,
      message: "Not found",
    });

    expect(isRetryableError(nonRetryable)).toBe(false);
    expect(isRetryableError(download404)).toBe(false);
  });

  it("retries on expanded status codes", () => {
    expect(isRetryableError(new DownloadError({ url: "https://api", statusCode: 401, message: "Unauthorized" }))).toBe(
      true,
    );
    expect(isRetryableError(new DownloadError({ url: "https://api", statusCode: 403, message: "Forbidden" }))).toBe(
      true,
    );
    expect(
      isRetryableError(new DownloadError({ url: "https://api", statusCode: 408, message: "Request Timeout" })),
    ).toBe(true);
    expect(isRetryableError(new DownloadError({ url: "https://api", statusCode: 409, message: "Conflict" }))).toBe(
      true,
    );
    expect(
      isRetryableError(new DownloadError({ url: "https://api", statusCode: 413, message: "Payload Too Large" })),
    ).toBe(true);
    expect(isRetryableError(new DownloadError({ url: "https://api", statusCode: 498, message: "Rate limit" }))).toBe(
      true,
    );
    expect(
      isRetryableError(new DownloadError({ url: "https://api", statusCode: 500, message: "Internal Server Error" })),
    ).toBe(true);
    expect(
      isRetryableError(new DownloadError({ url: "https://api", statusCode: 503, message: "Service Unavailable" })),
    ).toBe(true);
    expect(isRetryableError(new DownloadError({ url: "https://api", statusCode: 529, message: "Overloaded" }))).toBe(
      true,
    );
  });

  it("retries on expanded error message patterns", () => {
    expect(isRetryableError(new Error("API is overloaded"))).toBe(true);
    expect(isRetryableError(new Error("Service unavailable, try again later"))).toBe(true);
    expect(isRetryableError(new Error("Too many requests"))).toBe(true);
    expect(isRetryableError(new Error("TooManyRequests error"))).toBe(true);
    expect(isRetryableError(new Error("Internal server error occurred"))).toBe(true);
    expect(isRetryableError(new Error("Gateway timeout"))).toBe(true);
    expect(isRetryableError(new Error("rate_limit exceeded"))).toBe(true);
    expect(isRetryableError(new Error("ratelimit hit"))).toBe(true);
    expect(isRetryableError(new Error("Rate limit reached"))).toBe(true);
    expect(isRetryableError(new Error("Server at capacity"))).toBe(true);
    expect(isRetryableError(new Error("server_error: something went wrong"))).toBe(true);
    expect(isRetryableError(new Error("Request failed with status 429"))).toBe(true);
    expect(isRetryableError(new Error("HTTP 503 error"))).toBe(true);
  });

  it("does not retry on non-retryable error messages", () => {
    expect(isRetryableError(new Error("Invalid input format"))).toBe(false);
    expect(isRetryableError(new Error("Prompt context window exceeded"))).toBe(false);
    expect(isRetryableError(new Error("Server has exploded"))).toBe(false);
  });
});
