/**
 * Client-Side REST & Server-Sent Events (SSE) API Service
 *
 * Encapsulates all network communication between the React frontend (both the
 * Desktop Command Center and the Mobile Capture Route) and the Express backend.
 * Guarantees that the browser never imports `@google/genai` or touches API
 * secrets directly, while logging every client API call at INFO level.
 *
 * Use cases:
 * - Creating and hydrating QR-paired compiler sessions.
 * - Uploading smartphone or desktop video footage with real XHR progress tracking.
 * - Subscribing to live SSE pipeline telemetry and stage transitions.
 * - Configuring/testing external Cloud MuJoCo webhooks and polling Long Decode.
 */

import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type {
  CompilerSessionState,
  DeploymentBundle,
  ServerStreamEvent,
  UploadedVideoMeta,
  VideoSlot,
  WebhookConfig,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

/**
 * Safely parses an API `fetch` response as JSON, surfacing clear diagnostics
 * whenever a reverse proxy or auth gateway returns an unexpected HTML/text page
 * (such as an HTTP 200 auth interstitial) instead of JSON.
 */
async function parseApiJsonResponse<T extends { error?: string }>(
  response: Response,
  contextLabel: string,
): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  const rawText = await response.text();
  if (!rawText.trim()) {
    throw new Error(`${contextLabel} received an empty response (HTTP ${response.status}).`);
  }
  try {
    return JSON.parse(rawText) as T;
  } catch {
    logFunctionCall('parseApiJsonResponse.nonJsonError', {
      contextLabel,
      status: response.status,
      contentType,
      preview: rawText.slice(0, 140),
    });
    if (contentType.includes('text/html') || rawText.trimStart().startsWith('<')) {
      throw new Error(
        `${contextLabel} received an HTML page instead of JSON (HTTP ${response.status}). Refresh the preview tab to restore your session and retry.`,
      );
    }
    throw new Error(`${contextLabel} returned an unexpected response (HTTP ${response.status}).`);
  }
}

/**
 * Creates a fresh QR-pairable compiler session on the Express server.
 */
export async function createCompilerSession(
  taskGoal?: string,
  robotModel?: RobotModelId,
): Promise<CompilerSessionState> {
  logFunctionCall('createCompilerSession', { taskGoal, robotModel });
  const response = await fetch(APP_CONFIG.api.createSession, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskGoal, robotModel }),
  });
  const data = await parseApiJsonResponse<{ session?: CompilerSessionState; error?: string }>(
    response,
    'Create session',
  );
  if (!response.ok || !data.session) {
    throw new Error(data.error || `Failed to create session (HTTP ${response.status}).`);
  }
  return data.session;
}

/**
 * Fetches an existing session snapshot by ID (used on mobile capture pairing
 * and desktop page reloads).
 */
export async function fetchCompilerSession(sessionId: string): Promise<CompilerSessionState> {
  logFunctionCall('fetchCompilerSession', { sessionId });
  const response = await fetch(APP_CONFIG.api.getSession(sessionId), {
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{ session?: CompilerSessionState; error?: string }>(
    response,
    'Fetch session',
  );
  if (!response.ok || !data.session) {
    throw new Error(data.error || `Session '${sessionId}' could not be loaded.`);
  }
  return data.session;
}

/**
 * Uploads a video or frame image to a session using proxy-safe 1.5 MB binary
 * chunks over `window.fetch` (`init` -> `chunk` -> `complete`). Avoids
 * `XMLHttpRequest` proxy redirect issues and bypasses Cloud Run's 32 MB single-
 * request limit while reporting accurate percentage progress.
 */
export async function uploadSessionVideo(params: {
  sessionId: string;
  slot?: VideoSlot;
  file: File | Blob;
  filename: string;
  source: UploadedVideoMeta['source'];
  taskGoal?: string;
  autoStart?: boolean;
  onProgress?: (percent: number) => void;
}): Promise<{ slot: VideoSlot; video: UploadedVideoMeta; session: CompilerSessionState }> {
  const slot: VideoSlot = params.slot === 'secondary' ? 'secondary' : 'primary';
  logFunctionCall('uploadSessionVideo', {
    sessionId: params.sessionId,
    slot,
    filename: params.filename,
    sizeBytes: params.file.size,
    source: params.source,
    autoStart: Boolean(params.autoStart),
  });

  if (params.file.size <= 0) {
    throw new Error('Selected video file is empty (0 bytes).');
  }
  if (params.file.size > APP_CONFIG.upload.maxFileSizeBytes) {
    const maxMb = Math.round(APP_CONFIG.upload.maxFileSizeBytes / (1024 * 1024));
    throw new Error(`File exceeds the ${maxMb} MB maximum upload size.`);
  }

  params.onProgress?.(2);

  // Step 1: Initialize confined upload handle on the Express backend
  const initRes = await fetch(APP_CONFIG.api.uploadInit(params.sessionId), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      slot,
      filename: params.filename,
      mimeType: params.file.type || undefined,
      sizeBytes: params.file.size,
      source: params.source,
      taskGoal: params.taskGoal,
    }),
  });
  const initData = await parseApiJsonResponse<{
    ok?: boolean;
    uploadId?: string;
    error?: string;
  }>(initRes, 'Upload init');
  if (!initRes.ok || !initData.uploadId) {
    throw new Error(initData.error || `Upload initialization failed (HTTP ${initRes.status}).`);
  }

  // Step 2: Stream sequential 1.5 MB binary chunks via fetch
  const chunkSize = APP_CONFIG.upload.chunkSizeBytes;
  let offset = 0;
  let chunkIndex = 0;

  while (offset < params.file.size) {
    const end = Math.min(offset + chunkSize, params.file.size);
    const slice = params.file.slice(offset, end);

    const chunkRes = await fetch(
      APP_CONFIG.api.uploadChunk(params.sessionId, initData.uploadId, chunkIndex),
      {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: slice,
      },
    );
    const chunkData = await parseApiJsonResponse<{ ok?: boolean; error?: string }>(
      chunkRes,
      `Upload chunk ${chunkIndex + 1}`,
    );
    if (!chunkRes.ok || !chunkData.ok) {
      throw new Error(
        chunkData.error || `Upload chunk ${chunkIndex + 1} failed (HTTP ${chunkRes.status}).`,
      );
    }

    offset = end;
    chunkIndex += 1;
    params.onProgress?.(Math.min(96, Math.round((offset / params.file.size) * 95)));
  }

  // Step 3: Finalize upload & trigger SSE video_uploaded broadcast
  const completeRes = await fetch(APP_CONFIG.api.uploadComplete(params.sessionId), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      uploadId: initData.uploadId,
      autoStart: Boolean(params.autoStart),
    }),
  });
  const completeData = await parseApiJsonResponse<{
    ok?: boolean;
    slot?: VideoSlot;
    video?: UploadedVideoMeta;
    session?: CompilerSessionState;
    error?: string;
  }>(completeRes, 'Upload finalize');

  if (!completeRes.ok || !completeData.video || !completeData.session) {
    throw new Error(
      completeData.error || `Upload finalization failed (HTTP ${completeRes.status}).`,
    );
  }

  params.onProgress?.(100);
  return {
    slot: completeData.slot ?? slot,
    video: completeData.video,
    session: completeData.session,
  };
}

/**
 * Removes an uploaded video slot (typically the optional secondary view) from
 * the active session so the workspace returns to Single View mode.
 */
export async function removeSessionVideo(
  sessionId: string,
  slot: VideoSlot = 'secondary',
): Promise<CompilerSessionState> {
  logFunctionCall('removeSessionVideo', { sessionId, slot });
  const response = await fetch(APP_CONFIG.api.mediaFile(sessionId, slot), {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    session?: CompilerSessionState;
    error?: string;
  }>(response, 'Remove video');
  if (!response.ok || !data.session) {
    throw new Error(data.error || `Failed to remove ${slot} video (HTTP ${response.status}).`);
  }
  return data.session;
}

/**
 * Starts the 4-stage Gemini 4 Failure-to-Policy compilation pipeline on the server,
 * or mounts a virtual robot onto an already-mapped robot-free scene when
 * `forceSimulateRobot` is true.
 */
export async function triggerPipelineRun(params: {
  sessionId: string;
  taskGoal: string;
  robotModel: RobotModelId;
  rolloutDurationSec?: number;
  forceSimulateRobot?: boolean;
}): Promise<void> {
  logFunctionCall('triggerPipelineRun', params);
  const response = await fetch(APP_CONFIG.api.runPipeline(params.sessionId), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      taskGoal: params.taskGoal,
      robotModel: params.robotModel,
      rolloutDurationSec: params.rolloutDurationSec,
      forceSimulateRobot: Boolean(params.forceSimulateRobot),
    }),
  });
  const data = await parseApiJsonResponse<{ error?: string }>(response, 'Launch pipeline');
  if (!response.ok) {
    throw new Error(data.error || `Failed to launch pipeline (HTTP ${response.status}).`);
  }
}

/**
 * Cancels an active Gemini 4 pipeline run on the server and returns the
 * updated session state immediately.
 */
export async function cancelPipelineRun(sessionId: string): Promise<CompilerSessionState> {
  logFunctionCall('cancelPipelineRun', { sessionId });
  const response = await fetch(APP_CONFIG.api.cancelPipeline(sessionId), {
    method: 'POST',
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    session?: CompilerSessionState;
    error?: string;
  }>(response, 'Cancel pipeline');
  if (!response.ok || !data.session) {
    throw new Error(data.error || `Failed to cancel pipeline (HTTP ${response.status}).`);
  }
  return data.session;
}

/**
 * Saves external Cloud MuJoCo webhook configuration on the active session.
 */
export async function saveWebhookConfig(params: {
  sessionId: string;
  enabled: boolean;
  url: string;
  authToken?: string;
}): Promise<WebhookConfig> {
  logFunctionCall('saveWebhookConfig', {
    sessionId: params.sessionId,
    enabled: params.enabled,
    url: params.url,
    hasAuthToken: Boolean(params.authToken),
  });
  const response = await fetch(APP_CONFIG.api.configureWebhook(params.sessionId), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      enabled: params.enabled,
      url: params.url,
      authToken: params.authToken,
    }),
  });
  const data = await parseApiJsonResponse<{ webhook?: WebhookConfig; error?: string }>(
    response,
    'Save webhook',
  );
  if (!response.ok || !data.webhook) {
    throw new Error(data.error || `Webhook save failed (HTTP ${response.status}).`);
  }
  return data.webhook;
}

/**
 * Tests reachability of an external HTTPS Cloud MuJoCo webhook URL via the
 * server's SSRF-protected validator.
 */
export async function testWebhookConnection(
  url: string,
  authToken?: string,
): Promise<{ ok: boolean; latencyMs: number; message: string }> {
  logFunctionCall('testWebhookConnection', { url, hasAuthToken: Boolean(authToken) });
  const response = await fetch(APP_CONFIG.api.testWebhook, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, authToken }),
  });
  const data = await parseApiJsonResponse<{
    ok: boolean;
    latencyMs: number;
    message: string;
    error?: string;
  }>(response, 'Test webhook');
  return data;
}

/**
 * Polls the latest status and compiled content of the asynchronous Long Decode
 * deployment bundle.
 */
export async function pollDeploymentBundle(sessionId: string): Promise<DeploymentBundle | null> {
  logFunctionCall('pollDeploymentBundle', { sessionId });
  const response = await fetch(APP_CONFIG.api.bundleStatus(sessionId), {
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{ bundle?: DeploymentBundle | null; error?: string }>(
    response,
    'Poll bundle',
  );
  if (!response.ok) {
    throw new Error(data.error || `Bundle poll failed (HTTP ${response.status}).`);
  }
  return data.bundle ?? null;
}

/**
 * Opens an EventSource SSE connection for live session updates and returns a
 * cleanup function to close the stream on unmount.
 */
export function openSessionEventStream(
  sessionId: string,
  onEvent: (event: ServerStreamEvent) => void,
  onError?: () => void,
): () => void {
  logFunctionCall('openSessionEventStream', { sessionId });
  const es = new EventSource(APP_CONFIG.api.eventsStream(sessionId));

  es.onmessage = (messageEvent) => {
    try {
      const parsed = JSON.parse(messageEvent.data) as ServerStreamEvent;
      onEvent(parsed);
    } catch {
      // Ignore malformed keepalive chunks
    }
  };

  es.onerror = () => {
    if (onError) onError();
  };

  return () => {
    es.close();
  };
}
