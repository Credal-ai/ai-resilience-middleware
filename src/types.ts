import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2Middleware,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider";

export type ResilienceAttemptPhase = "same_model_retry" | "cross_provider_fallback";

export type ResilienceAttemptDetails = {
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
};

export type IntermediateStateWithError = {
  errorDuringStreamAttempt?: Error;
  accumulatedStreamChunks: LanguageModelV2StreamPart[];
};

export type FallbackModelConfig = {
  modelId: string;
  createModel: () => LanguageModelV2;
};

export type ResilienceLogger = {
  info: (message: string, meta?: Record<string, unknown>) => void;
  warn: (message: string, meta?: Record<string, unknown>) => void;
  error: (message: string, meta?: Record<string, unknown>) => void;
};

export type ResilienceMiddlewareConfig = {
  fallbackModels: FallbackModelConfig[];
  maxSameModelRetries?: number;
  initialRetryDelayMs?: number;
  maxRetryDelayMs?: number;
  shouldApply?: (params: LanguageModelV2CallOptions) => boolean;
  isFallbackEnabled?: (params: LanguageModelV2CallOptions, modelId: string) => boolean;
  logger?: ResilienceLogger;
  onAttemptFailed?: (details: ResilienceAttemptDetails) => void | Promise<void>;
};

export type CreateResilienceMiddleware = (config: ResilienceMiddlewareConfig) => LanguageModelV2Middleware;
