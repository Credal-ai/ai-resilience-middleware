import type {
  LanguageModelV2Message,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
  LanguageModelV2ToolCallPart,
  LanguageModelV2ToolResultOutput,
  LanguageModelV2ToolResultPart,
  SharedV2ProviderOptions,
} from "@ai-sdk/provider";
import { APICallError, DownloadError, type JSONValue } from "ai";

const DEFAULT_INITIAL_RETRY_DELAY_MS = 1000;
const DEFAULT_MAX_RETRY_DELAY_MS = 8000;
const EXPONENTIAL_BACKOFF_MULTIPLIER = 2;

export function extractPromptText(prompt: LanguageModelV2Prompt): string {
  const parts: string[] = [];

  for (const message of prompt) {
    if (message.role === "system") {
      parts.push(`${message.content}`);
    } else if (message.role === "user") {
      for (const part of message.content) {
        if (part.type === "text") {
          parts.push(`${part.text}`);
        }
      }
    } else if (message.role === "assistant") {
      for (const part of message.content) {
        if (part.type === "text" || part.type === "reasoning") {
          parts.push(`${part.text}`);
        }
      }
    }
  }

  return parts.join("\n");
}

export function extractCompletionPartialText(chunks: LanguageModelV2StreamPart[]): string {
  const rawTextFromModelStream: string[] = [];

  for (const chunk of chunks) {
    if (chunk.type === "text-delta") {
      rawTextFromModelStream.push(chunk.delta);
    } else if (chunk.type === "reasoning-delta") {
      rawTextFromModelStream.push(chunk.delta);
    }
  }

  return rawTextFromModelStream.join("");
}

type AssistantContentPart = Extract<LanguageModelV2Message, { role: "assistant" }>["content"][number];
type ToolResultChunk = Extract<LanguageModelV2StreamPart, { type: "tool-result" }>;
type CurrentSegment = { type: "text" | "reasoning"; id: string; deltas: string[] } | null;

function finalizeSegment(segment: CurrentSegment, orderedContent: AssistantContentPart[]): CurrentSegment {
  if (segment && segment.deltas.length > 0) {
    const segmentText = segment.deltas.join("");
    orderedContent.push({ type: segment.type, text: segmentText });
  }
  return null;
}

function safeParseJson(value: unknown): unknown {
  if (typeof value !== "string") {
    return value ?? {};
  }

  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function mapToolResultOutput(chunk: ToolResultChunk): LanguageModelV2ToolResultOutput {
  const toolResultOutput = chunk.result;
  if (toolResultOutput && typeof toolResultOutput === "object" && "type" in toolResultOutput) {
    return toolResultOutput as LanguageModelV2ToolResultOutput;
  }
  if (toolResultOutput !== undefined) {
    return { type: "json", value: toolResultOutput as JSONValue };
  }
  return { type: "json", value: null };
}

function buildModelMessagesFromPartialChunks(chunks: LanguageModelV2StreamPart[]): LanguageModelV2Message[] {
  const messages: LanguageModelV2Message[] = [];
  const pendingToolCallsById = new Map<string, LanguageModelV2ToolCallPart>();
  let currentSegment: CurrentSegment = null;
  let assistantParts: AssistantContentPart[] = [];

  const flushAssistantParts = () => {
    currentSegment = finalizeSegment(currentSegment, assistantParts);
    if (assistantParts.length > 0) {
      messages.push({ role: "assistant", content: assistantParts });
      assistantParts = [];
    }
  };

  for (const chunk of chunks) {
    switch (chunk.type) {
      case "text-start": {
        currentSegment = finalizeSegment(currentSegment, assistantParts);
        currentSegment = { type: "text", id: chunk.id, deltas: [] };
        break;
      }
      case "text-delta": {
        if (currentSegment && currentSegment.id === chunk.id) {
          currentSegment.deltas.push(chunk.delta);
        } else {
          currentSegment = finalizeSegment(currentSegment, assistantParts);
          currentSegment = { type: "text", id: chunk.id, deltas: [chunk.delta] };
        }
        break;
      }
      case "text-end": {
        currentSegment = finalizeSegment(currentSegment, assistantParts);
        break;
      }
      case "reasoning-start": {
        currentSegment = finalizeSegment(currentSegment, assistantParts);
        currentSegment = { type: "reasoning", id: chunk.id, deltas: [] };
        break;
      }
      case "reasoning-delta": {
        if (currentSegment && currentSegment.id === chunk.id) {
          currentSegment.deltas.push(chunk.delta);
        } else {
          currentSegment = finalizeSegment(currentSegment, assistantParts);
          currentSegment = { type: "reasoning", id: chunk.id, deltas: [chunk.delta] };
        }
        break;
      }
      case "reasoning-end": {
        currentSegment = finalizeSegment(currentSegment, assistantParts);
        break;
      }
      case "tool-call": {
        const toolCallPart: LanguageModelV2ToolCallPart = {
          type: "tool-call",
          toolCallId: chunk.toolCallId,
          toolName: chunk.toolName,
          input: safeParseJson(chunk.input),
          providerExecuted: chunk.providerExecuted,
          providerOptions: chunk.providerMetadata as SharedV2ProviderOptions,
        };
        if (chunk.providerExecuted) {
          pendingToolCallsById.set(chunk.toolCallId, toolCallPart);
        } else {
          currentSegment = finalizeSegment(currentSegment, assistantParts);
          assistantParts.push(toolCallPart);
        }
        break;
      }
      case "tool-result": {
        const toolResultPart: LanguageModelV2ToolResultPart = {
          type: "tool-result",
          toolCallId: chunk.toolCallId,
          toolName: chunk.toolName,
          output: mapToolResultOutput(chunk),
          providerOptions: chunk.providerMetadata as SharedV2ProviderOptions,
        };
        if (chunk.providerExecuted) {
          const pendingCall = pendingToolCallsById.get(chunk.toolCallId);
          if (!pendingCall) {
            break;
          }
          pendingToolCallsById.delete(chunk.toolCallId);
          flushAssistantParts();
          messages.push({ role: "assistant", content: [pendingCall] });
          messages.push({ role: "tool", content: [toolResultPart] });
        } else {
          currentSegment = finalizeSegment(currentSegment, assistantParts);
          flushAssistantParts();
          messages.push({ role: "tool", content: [toolResultPart] });
        }
        break;
      }
    }
  }

  flushAssistantParts();
  return messages;
}

export function reconstructPromptWithPartialChunks(
  originalPrompt: LanguageModelV2Prompt,
  accumulatedChunks: LanguageModelV2StreamPart[],
): LanguageModelV2Prompt {
  const assistantAndToolResultsContent = buildModelMessagesFromPartialChunks(accumulatedChunks);
  const originalPromptWithoutEmptyAssistantMessages = originalPrompt.filter(message => {
    if (message.role === "assistant") {
      const hasContent = message.content.some(part => {
        if (part.type === "text" || part.type === "reasoning") {
          return part.text.trim().length > 0;
        }
        return true;
      });
      return hasContent;
    }
    return true;
  });

  return [...originalPromptWithoutEmptyAssistantMessages, ...assistantAndToolResultsContent];
}

export function normalizeStreamError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }
  if (typeof error === "string") {
    return new Error(error);
  }
  try {
    return new Error(JSON.stringify(error));
  } catch {
    return new Error("Unknown stream error");
  }
}

export function getRetryDelay(
  attemptNumber: number,
  initialRetryDelayMs = DEFAULT_INITIAL_RETRY_DELAY_MS,
  maxRetryDelayMs = DEFAULT_MAX_RETRY_DELAY_MS,
): number {
  const delay = initialRetryDelayMs * Math.pow(EXPONENTIAL_BACKOFF_MULTIPLIER, attemptNumber - 1);
  return Math.min(delay, maxRetryDelayMs);
}

const retryableStatusCodes = [401, 403, 408, 409, 413, 429, 498];
const retryableErrorPatterns = [
  "overloaded",
  "service unavailable",
  "bad gateway",
  "too many requests",
  "toomanyrequests",
  "internal server error",
  "gateway timeout",
  "rate_limit",
  "ratelimit",
  "rate limit",
  "wrong-key",
  "unexpected",
  "capacity",
  "timeout",
  "server_error",
  "connection",
  "request_too_large",
  "network",
  "429",
  "500",
  "502",
  "503",
  "504",
];

export function isRetryableError(error: unknown): boolean {
  if (error instanceof APICallError) {
    return error.isRetryable;
  }

  if (error instanceof Error && "isRetryable" in error && error.isRetryable) {
    return true;
  }

  if (error instanceof DownloadError) {
    const statusCode = error.statusCode ?? 0;
    if (retryableStatusCodes.includes(statusCode) || statusCode >= 500) {
      return true;
    }
  }

  const msg = error instanceof Error ? error.message?.toLowerCase() || "" : "";
  if (retryableErrorPatterns.some(pattern => msg.includes(pattern))) {
    return true;
  }
  if (isSocketError(error)) {
    return true;
  }

  return false;
}

const modelUnavailablePatterns = [
  "not_found",
  "not found",
  "model not found",
  "retired",
  "deprecated",
  "decommissioned",
  "does not exist",
  "no longer available",
  "invalid model",
  "unknown model",
  "unsupported model",
];

function isModelUnavailableError(error: unknown): boolean {
  if (error instanceof APICallError) {
    if (error.statusCode === 404) {
      return true;
    }
    const msg = (error.message?.toLowerCase() ?? "") + (error.responseBody?.toLowerCase() ?? "");
    return modelUnavailablePatterns.some(pattern => msg.includes(pattern));
  }

  const msg = error instanceof Error ? (error.message?.toLowerCase() ?? "") : "";
  return modelUnavailablePatterns.some(pattern => msg.includes(pattern));
}

export function shouldAttemptCrossProviderFallback(error: unknown): boolean {
  return isRetryableError(error) || isModelUnavailableError(error);
}

function isSocketError(error: unknown): error is Error {
  if (!(error instanceof Error)) return false;
  const cause = error.cause;
  if (cause == null || typeof cause !== "object") return false;
  if (!("name" in cause)) return false;
  return cause.name === "SocketError";
}
