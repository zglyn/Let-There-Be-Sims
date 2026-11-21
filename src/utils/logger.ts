/**
 * Structured Info Logger & GenAI Payload Sanitizer
 *
 * Provides uniform INFO-level logging across both server and client modules for:
 * 1. Every function invocation together with its sanitized parameters.
 * 2. Every Gemini Interactions API / GenAI call including model name, prompt,
 *    configuration, and returned outputs.
 *
 * Inline binary payloads (base64 videos, images, audio buffers, or oversized
 * data URIs) are automatically stripped and replaced with compact byte-length
 * placeholders so logs stay readable and never leak raw binary blobs or secrets.
 *
 * Use cases:
 * - Auditing multi-model Gemini 4 calls (`gemini-4-carbon`, `gemini-4-hydrogen`,
 *   `gemini-4-neon`) and `previous_interaction_id` chains.
 * - Tracing function entry points across Express routes, physics rollouts, and
 *   React command-center actions.
 */

const INLINE_KEYS = new Set([
  'data',
  'inlineData',
  'inline_data',
  'video_b64',
  'preview_videos_b64',
  'audioData',
  'base64',
  'buffer',
  'authToken',
  'apiKey',
  'authorization',
]);

const MAX_STRING_LOG_LENGTH = 1400;

/**
 * Recursively sanitizes any arbitrary value for structured info logging by
 * stripping inline binary data, base64 URIs, secrets, and circular references.
 */
export function sanitizeForLogging(value: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === 'string') {
    if (value.startsWith('data:') && value.includes(';base64,')) {
      const header = value.slice(0, value.indexOf(';base64,') + 8);
      return `${header}[STRIPPED_INLINE_DATA:${value.length}_chars]`;
    }
    if (value.length > MAX_STRING_LOG_LENGTH) {
      return `${value.slice(0, MAX_STRING_LOG_LENGTH)}... [TRUNCATED:${value.length}_chars]`;
    }
    return value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'function') {
    return `[Function:${value.name || 'anonymous'}]`;
  }
  if (typeof Uint8Array !== 'undefined' && value instanceof Uint8Array) {
    return `[STRIPPED_BINARY:${value.byteLength}_bytes]`;
  }
  if (depth > 5) {
    return '[MAX_DEPTH]';
  }
  if (Array.isArray(value)) {
    const sliced = value.slice(0, 25).map((item) => sanitizeForLogging(item, depth + 1, seen));
    if (value.length > 25) {
      sliced.push(`[+${value.length - 25}_more_items]`);
    }
    return sliced;
  }
  if (typeof value === 'object') {
    if (seen.has(value)) {
      return '[CIRCULAR_REF]';
    }
    seen.add(value);
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (INLINE_KEYS.has(k)) {
        if (typeof v === 'string') {
          out[k] = `[STRIPPED_INLINE_DATA:${v.length}_chars]`;
        } else if (Array.isArray(v)) {
          out[k] = `[STRIPPED_INLINE_ARRAY:${v.length}_items]`;
        } else {
          out[k] = '[STRIPPED_INLINE_DATA]';
        }
      } else {
        out[k] = sanitizeForLogging(v, depth + 1, seen);
      }
    }
    return out;
  }
  return String(value);
}

/**
 * Logs a function invocation at INFO level together with its sanitized arguments.
 */
export function logFunctionCall(functionName: string, params: Record<string, unknown> = {}): void {
  const timestamp = new Date().toISOString();
  console.info(`[INFO][${timestamp}][FN] ${functionName}`, sanitizeForLogging(params));
}

/**
 * Logs a GenAI API request and response at INFO level with model, prompt,
 * configuration, and outputs while stripping inline binary payloads.
 */
export function logGenAiCall(details: {
  stage: string;
  model: string;
  previousInteractionId?: string | null;
  prompt: unknown;
  config?: unknown;
  output?: unknown;
  interactionId?: string;
  durationMs?: number;
  error?: string;
}): void {
  const timestamp = new Date().toISOString();
  console.info(
    `[INFO][${timestamp}][GENAI] stage=${details.stage} model=${details.model}`,
    sanitizeForLogging({
      interactionId: details.interactionId ?? null,
      previousInteractionId: details.previousInteractionId ?? null,
      durationMs: details.durationMs ?? null,
      prompt: details.prompt,
      config: details.config,
      output: details.output,
      error: details.error ?? null,
    }),
  );
}
