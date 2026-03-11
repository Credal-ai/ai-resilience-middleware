# ai-resilience-middleware

Resilience middleware for the [Vercel AI SDK](https://sdk.vercel.ai/docs) that adds automatic retry and cross-provider fallback to streaming LLM requests.

## Features

- **Same-model retries** with exponential backoff
- **Cross-provider fallback chain** — if all retries fail, transparently switch to a different model/provider
- **Mid-stream reconstruction** — partial responses are folded back into the prompt so the fallback model can continue where the original left off
- **Zero buffering** — chunks are forwarded to the consumer in real-time
- **Provider-agnostic** — works with any `LanguageModelV2` (OpenAI, Anthropic, Google, etc.)

## Install

```bash
npm install ai-resilience-middleware
```

Peer dependencies: `@ai-sdk/provider`, `ai`

## Quick Start

```typescript
import { createResilienceMiddleware } from "ai-resilience-middleware";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

const resilience = createResilienceMiddleware({
  fallbackModels: [
    {
      modelId: "claude-haiku-4-5-20251001",
      createModel: () =>
        createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY! })
          .languageModel("claude-haiku-4-5-20251001"),
    },
    {
      modelId: "gemini-2.5-flash",
      createModel: () =>
        createGoogleGenerativeAI({ apiKey: process.env.GEMINI_API_KEY! })
          .languageModel("gemini-2.5-flash"),
    },
  ],
});
```

Then register it as AI SDK middleware via `wrapLanguageModel` or your provider setup.

## Configuration

```typescript
type ResilienceMiddlewareConfig = {
  /** Ordered fallback chain — attempted in sequence until one succeeds. */
  fallbackModels: Array<{
    modelId: string;
    createModel: () => LanguageModelV2;
  }>;

  /** Max same-model retries before falling back. Default: 1 */
  maxSameModelRetries?: number;

  /** Initial retry delay in ms. Default: 1000 */
  initialRetryDelayMs?: number;

  /** Max retry delay in ms (caps exponential backoff). Default: 8000 */
  maxRetryDelayMs?: number;

  /** Gate function — return false to skip resilience for a request. Default: always apply */
  shouldApply?: (params: LanguageModelV2CallOptions) => boolean;

  /** Per-request check for whether a specific fallback model is enabled. Default: all enabled */
  isFallbackEnabled?: (params: LanguageModelV2CallOptions, modelId: string) => boolean;

  /** Structured logger. Default: silent (no-op) */
  logger?: ResilienceLogger;

  /** Fire-and-forget callback on every failed attempt — use for audit logging, metrics, etc. */
  onAttemptFailed?: (details: ResilienceAttemptDetails) => void | Promise<void>;
};
```

## How It Works

1. The primary model streams normally. If an error chunk or connection failure occurs mid-stream, the middleware catches it.
2. It retries the **same model** up to `maxSameModelRetries` times with exponential backoff. On each retry the prompt is reconstructed to include the partial response received so far.
3. If all same-model retries fail, it walks the **fallback chain** in order, reconstructing the prompt each time.
4. If all fallbacks also fail, the original error propagates to the consumer.

## License

MIT
