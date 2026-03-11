import type { ResilienceLogger } from "./types";

export function createDefaultLogger(): ResilienceLogger {
  return {
    info: (_message, _meta) => {},
    warn: (_message, _meta) => {},
    error: (_message, _meta) => {},
  };
}
