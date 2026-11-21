/**
 * Server-Side Gemini 4 Interactions API Client & Media Helper
 *
 * Wraps the `@google/genai` SDK exclusively on the Express backend so
 * `GEMINI_API_KEY` never leaves the server. Provides:
 * - Singleton `GoogleGenAI` client initialization.
 * - Automatic media preparation: compresses large videos (> 2.5 MB) to a fast
 *   1 FPS / 480p full-timeline analysis proxy via `videoTranscoder.ts` and uses
 *   the Gemini Files API for anything above 2 MB (inline base64 only for small
 *   clips/images), preventing bloated HTTP POST bodies and long upload stalls.
 * - Standardized `createLoggedInteraction` wrapper supporting BOTH synchronous
 *   turns and resilient `background: true` + `waitForCompletion: true` polling
 *   (immune to long-lived HTTP/2 socket drops), with hard per-stage timeouts
 *   and `AbortSignal` cancellation so no stage can ever hang indefinitely.
 * - Safe background interaction polling via `ai.interactions.get` and INFO-level
 *   GenAI logging with inline binary data stripped.
 *
 * Use cases:
 * - Chaining `gemini-4-carbon` -> `gemini-4-hydrogen` -> `gemini-4-neon` via
 *   `previous_interaction_id` with explicit `thinking_level` escalation.
 * - Polling long-running multimodal Stage 1 / Stage 3 turns and asynchronous
 *   Stage 4 Long Decode deployment bundles.
 */

import fs from 'fs';
import { GoogleGenAI } from '@google/genai';
import { APP_CONFIG } from '../config/appConfig';
import { logFunctionCall, logGenAiCall } from '../utils/logger';
import { buildAnalysisVideoProxy } from './videoTranscoder';

let cachedClient: GoogleGenAI | null = null;

/**
 * Returns the shared server-side GoogleGenAI SDK client instance.
 */
export function getGeminiClient(): GoogleGenAI {
  logFunctionCall('getGeminiClient');
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is not configured on the server. Select a paid API key in AI Studio Settings > Secrets.',
    );
  }
  if (!cachedClient) {
    cachedClient = new GoogleGenAI({ apiKey });
  }
  return cachedClient;
}

/**
 * Sleeps for `ms` milliseconds while respecting an optional `AbortSignal`.
 */
function sleepWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error('Pipeline cancelled by user.'));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new Error('Pipeline cancelled by user.'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Races an async promise against a hard timeout and an optional `AbortSignal`
 * so stalled network sockets always reject with a clear diagnostic error.
 */
async function withTimeoutAndAbort<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string,
  signal?: AbortSignal,
): Promise<T> {
  if (signal?.aborted) {
    throw new Error('Pipeline cancelled by user.');
  }

  let timer: ReturnType<typeof setTimeout> | null = null;
  let onAbort: (() => void) | null = null;

  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
        if (signal) {
          onAbort = () => reject(new Error('Pipeline cancelled by user.'));
          signal.addEventListener('abort', onAbort, { once: true });
        }
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
    if (signal && onAbort) signal.removeEventListener('abort', onAbort);
  }
}

export interface PreparedMediaPart {
  type: 'video' | 'image';
  mime_type: string;
  uri?: string;
  data?: string;
}

/**
 * Prepares an uploaded video or image file for the Gemini Interactions API:
 * - Compresses large videos (> 2.5 MB) to a fast 1 FPS / 480p analysis proxy
 *   covering the entire clip duration.
 * - Uses the Gemini Files API for files > 2 MB (or inline base64 for small
 *   clips/images), then cleans up any temporary proxy file.
 */
export async function prepareMediaForInteractions(
  filePath: string,
  mimeType: string,
  signal?: AbortSignal,
): Promise<PreparedMediaPart> {
  logFunctionCall('prepareMediaForInteractions', { filePath, mimeType });
  if (signal?.aborted) {
    throw new Error('Pipeline cancelled by user.');
  }

  const proxy = await buildAnalysisVideoProxy(filePath, mimeType);
  const mediaType: 'video' | 'image' = proxy.mimeType.startsWith('image/') ? 'image' : 'video';

  try {
    if (proxy.proxyBytes <= APP_CONFIG.upload.inlineBase64LimitBytes) {
      const base64Data = fs.readFileSync(proxy.filePath).toString('base64');
      return {
        type: mediaType,
        mime_type: proxy.mimeType,
        data: base64Data,
      };
    }

    const ai = getGeminiClient();
    const uploaded = await withTimeoutAndAbort(
      ai.files.upload({
        file: proxy.filePath,
        config: { mimeType: proxy.mimeType },
      }),
      60_000,
      'Uploading video to Gemini Files API timed out after 60s. Retry or use a shorter clip.',
      signal,
    );

    let current = uploaded;
    const maxWaitMs = 90_000;
    const start = Date.now();
    while (current.state === 'PROCESSING' && Date.now() - start < maxWaitMs) {
      await sleepWithAbort(2000, signal);
      if (!current.name) break;
      current = await withTimeoutAndAbort(
        ai.files.get({ name: current.name }),
        20_000,
        'Gemini Files API status check timed out.',
        signal,
      );
    }

    if (current.state === 'FAILED') {
      throw new Error('Gemini Files API failed to process the uploaded video.');
    }
    if (current.state === 'PROCESSING') {
      throw new Error(
        'Gemini Files API took too long (>90s) to finish video indexing. Try a shorter clip.',
      );
    }

    return {
      type: mediaType,
      mime_type: current.mimeType ?? proxy.mimeType,
      uri: current.uri,
    };
  } finally {
    proxy.cleanup();
  }
}

export interface ExtractedInteractionOutput {
  id: string;
  status: string;
  text: string;
  thoughtSummary: string;
  durationMs: number;
}

/**
 * Extracts combined model output text and chain-of-thought summaries from an
 * Interactions API response object safely across single and multi-step turns.
 */
export function extractInteractionContent(interaction: {
  id?: string;
  status?: string;
  output_text?: string | null;
  steps?: Array<{
    type?: string;
    content?: Array<{ type?: string; text?: string }>;
    summary?: Array<{ type?: string; text?: string }>;
  }>;
}): { text: string; thoughtSummary: string } {
  let combinedText = '';
  const thoughts: string[] = [];

  if (Array.isArray(interaction.steps)) {
    for (const step of interaction.steps) {
      if (step.type === 'thought' && Array.isArray(step.summary)) {
        const snippet = step.summary
          .map((part) => (typeof part.text === 'string' ? part.text : ''))
          .join('')
          .trim();
        if (snippet) thoughts.push(snippet);
      }
      if (step.type === 'model_output' && Array.isArray(step.content)) {
        for (const part of step.content) {
          if (part.type === 'text' && typeof part.text === 'string') {
            combinedText += part.text;
          }
        }
      }
    }
  }

  const finalText = combinedText.trim() || (interaction.output_text ?? '').trim();
  return {
    text: finalText,
    thoughtSummary: thoughts.join('\n\n'),
  };
}

const RAW_CONTENT_TYPES = new Set(['text', 'image', 'video', 'audio', 'document']);

/**
 * Normalizes `input` for the Gemini Interactions API `v1` wire format.
 * Plain strings pass through directly; arrays of raw multimodal content parts
 * (e.g. `video` + `text`) are wrapped in a `user_input` step object as required
 * by `v1`.
 */
export function normalizeInteractionInput(input: unknown): unknown {
  if (typeof input === 'string') {
    return input;
  }
  if (Array.isArray(input) && input.length > 0) {
    const first = input[0] as { type?: string } | null;
    if (
      first &&
      typeof first === 'object' &&
      typeof first.type === 'string' &&
      RAW_CONTENT_TYPES.has(first.type)
    ) {
      return [
        {
          type: 'user_input',
          content: input,
        },
      ];
    }
  }
  return input;
}

/**
 * Executes a Gemini 4 `ai.interactions.create` call with full structured INFO
 * logging (model, prompt, config, outputs, inline data stripped), hard timeout
 * protection, `AbortSignal` cancellation, and optional resilient background
 * polling (`background: true` + `waitForCompletion: true`).
 */
export async function createLoggedInteraction(params: {
  stage: string;
  model: string;
  input: unknown;
  systemInstruction?: string;
  previousInteractionId?: string | null;
  thinkingLevel: 'low' | 'medium' | 'high' | 'xhigh';
  responseFormat?: Record<string, unknown>;
  background?: boolean;
  waitForCompletion?: boolean;
  timeoutMs?: number;
  signal?: AbortSignal;
  onPollHeartbeat?: (status: string, elapsedMs: number) => void;
}): Promise<ExtractedInteractionOutput> {
  const ai = getGeminiClient();
  const startedAt = Date.now();
  const normalizedInput = normalizeInteractionInput(params.input);
  const timeoutMs = params.timeoutMs ?? APP_CONFIG.timeouts.carbonStageMs;

  const requestPayload: Record<string, unknown> = {
    model: params.model,
    input: normalizedInput,
    store: true,
    generation_config: {
      thinking_level: params.thinkingLevel,
    },
  };

  if (params.systemInstruction) {
    requestPayload.system_instruction = params.systemInstruction;
  }
  if (params.previousInteractionId) {
    requestPayload.previous_interaction_id = params.previousInteractionId;
  }
  if (params.responseFormat) {
    requestPayload.response_format = params.responseFormat;
  }
  if (params.background) {
    requestPayload.background = true;
  }

  try {
    const createTimeoutMs = params.background
      ? Math.min(timeoutMs, APP_CONFIG.timeouts.createHandshakeMs)
      : timeoutMs;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let interaction = await withTimeoutAndAbort<any>(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (ai.interactions.create as any)(requestPayload),
      createTimeoutMs,
      `${params.model} (${params.stage}) timed out after ${Math.round(createTimeoutMs / 1000)}s. Retry or use a shorter clip / single view.`,
      params.signal,
    );

    // If dispatched in background mode and caller wants the finished output,
    // poll safely every 3s so long reasoning turns never hang on a dead HTTP socket.
    if (params.background && params.waitForCompletion) {
      const pollIntervalMs = APP_CONFIG.timeouts.stagePollIntervalMs;
      while (
        interaction.status !== 'completed' &&
        interaction.status !== 'failed' &&
        interaction.status !== 'cancelled'
      ) {
        const elapsedMs = Date.now() - startedAt;
        if (elapsedMs >= timeoutMs) {
          throw new Error(
            `${params.model} (${params.stage}) exceeded the ${Math.round(timeoutMs / 1000)}s stage time limit (status: ${interaction.status ?? 'in_progress'}). Click Analyze & Verify Policy to retry, or try Single View / a shorter clip.`,
          );
        }
        params.onPollHeartbeat?.(String(interaction.status ?? 'in_progress'), elapsedMs);
        await sleepWithAbort(pollIntervalMs, params.signal);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        interaction = await withTimeoutAndAbort<any>(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (ai.interactions.get as any)(interaction.id),
          25_000,
          `Polling ${params.model} (${params.stage}) timed out while checking status.`,
          params.signal,
        );
      }

      if (interaction.status === 'failed' || interaction.status === 'cancelled') {
        const errDetail =
          typeof interaction.error?.message === 'string'
            ? interaction.error.message
            : `Interaction ended with status '${interaction.status}'.`;
        throw new Error(`${params.model} (${params.stage}) failed: ${errDetail}`);
      }
    }

    const durationMs = Date.now() - startedAt;
    const { text, thoughtSummary } = extractInteractionContent(interaction);

    logGenAiCall({
      stage: params.stage,
      model: params.model,
      previousInteractionId: params.previousInteractionId ?? null,
      prompt: params.input,
      config: {
        thinking_level: params.thinkingLevel,
        hasResponseFormat: Boolean(params.responseFormat),
        background: Boolean(params.background),
        waitForCompletion: Boolean(params.waitForCompletion),
      },
      interactionId: interaction.id,
      durationMs,
      output: {
        status: interaction.status ?? 'completed',
        thoughtSummary,
        text,
      },
    });

    return {
      id: interaction.id,
      status: interaction.status ?? 'completed',
      text,
      thoughtSummary,
      durationMs,
    };
  } catch (err) {
    const durationMs = Date.now() - startedAt;
    const message = err instanceof Error ? err.message : String(err);
    logGenAiCall({
      stage: params.stage,
      model: params.model,
      previousInteractionId: params.previousInteractionId ?? null,
      prompt: params.input,
      config: {
        thinking_level: params.thinkingLevel,
        background: Boolean(params.background),
      },
      durationMs,
      error: message,
    });
    throw err;
  }
}

/**
 * Fetches the latest status and outputs of an asynchronous background Gemini
 * interaction by ID and logs the retrieved state.
 */
export async function getLoggedInteraction(interactionId: string): Promise<ExtractedInteractionOutput> {
  logFunctionCall('getLoggedInteraction', { interactionId });
  const ai = getGeminiClient();
  const startedAt = Date.now();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const interaction = await withTimeoutAndAbort<any>(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (ai.interactions.get as any)(interactionId),
    25_000,
    'Timed out polling Long Decode bundle status.',
  );
  const durationMs = Date.now() - startedAt;
  const { text, thoughtSummary } = extractInteractionContent(interaction);

  logGenAiCall({
    stage: 'poll_background_interaction',
    model: APP_CONFIG.models.bundleCompiler.id,
    interactionId,
    prompt: `GET /interactions/${interactionId}`,
    durationMs,
    output: { status: interaction.status, textLength: text.length },
  });

  return {
    id: interaction.id ?? interactionId,
    status: interaction.status ?? 'in_progress',
    text,
    thoughtSummary,
    durationMs,
  };
}

/**
 * Extracts and parses JSON safely from a model text output, handling optional
 * fenced markdown code blocks cleanly.
 */
export function parseStructuredJson<T>(rawText: string): T {
  logFunctionCall('parseStructuredJson', { textLength: rawText.length });
  const trimmed = rawText.trim();
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const match =
      trimmed.match(/```json\s*([\s\S]*?)\s*```/i) ||
      trimmed.match(/(\{[\s\S]*\})/);
    if (!match) {
      throw new Error('Model response did not contain a valid JSON payload.');
    }
    return JSON.parse(match[1]) as T;
  }
}
