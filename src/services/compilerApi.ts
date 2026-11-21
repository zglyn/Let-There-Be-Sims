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
  GymMjxExportConfig,
  GymMjxPackageResponse,
  LeRobotDatasetConfig,
  LeRobotDatasetPreviewResponse,
  MobileSensorCapture,
  RegressionCiArtifacts,
  RegressionRunMode,
  RegressionSuiteRunRecord,
  RegressionSuiteSnapshot,
  RegressionTestCase,
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
  mobileSensors?: MobileSensorCapture | null;
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
    hasMobileSensors: Boolean(params.mobileSensors),
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
      mobileSensors: params.mobileSensors ?? null,
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

/**
 * Fetches all saved Failure-to-Regression-Suite CI test cases and build history.
 */
export async function fetchRegressionSuite(): Promise<RegressionSuiteSnapshot> {
  logFunctionCall('fetchRegressionSuite');
  const response = await fetch(APP_CONFIG.api.regressionCases, {
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<RegressionSuiteSnapshot & { error?: string }>(
    response,
    'Fetch regression suite',
  );
  if (!response.ok) {
    throw new Error(data.error || `Failed to load regression suite (HTTP ${response.status}).`);
  }
  return {
    cases: Array.isArray(data.cases) ? data.cases : [],
    runs: Array.isArray(data.runs) ? data.runs : [],
  };
}

/**
 * Promotes the active session's compiled failure/scene into a permanent
 * regression test case persisted on the server.
 */
export async function saveSessionToRegressionSuite(params: {
  sessionId: string;
  name?: string;
  tags?: string[];
}): Promise<{ testCase: RegressionTestCase; suite: RegressionSuiteSnapshot }> {
  logFunctionCall('saveSessionToRegressionSuite', params);
  const response = await fetch(APP_CONFIG.api.regressionCases, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    testCase?: RegressionTestCase;
    suite?: RegressionSuiteSnapshot;
    error?: string;
  }>(response, 'Save regression test case');
  if (!response.ok || !data.testCase || !data.suite) {
    throw new Error(data.error || `Failed to save regression test case (HTTP ${response.status}).`);
  }
  return { testCase: data.testCase, suite: data.suite };
}

/**
 * Removes a saved regression test case from the CI suite.
 */
export async function deleteRegressionTestCase(caseId: string): Promise<RegressionSuiteSnapshot> {
  logFunctionCall('deleteRegressionTestCase', { caseId });
  const response = await fetch(APP_CONFIG.api.regressionCase(caseId), {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    suite?: RegressionSuiteSnapshot;
    error?: string;
  }>(response, 'Delete regression test case');
  if (!response.ok || !data.suite) {
    throw new Error(data.error || `Failed to delete regression case (HTTP ${response.status}).`);
  }
  return data.suite;
}

/**
 * Executes the entire Regression Suite CI gate across every saved failure case.
 */
export async function runRegressionSuiteCiApi(params: {
  mode: RegressionRunMode;
  sessionId?: string;
  policyLabel?: string;
}): Promise<{ run: RegressionSuiteRunRecord; suite: RegressionSuiteSnapshot }> {
  logFunctionCall('runRegressionSuiteCiApi', params);
  const response = await fetch(APP_CONFIG.api.regressionRun, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    run?: RegressionSuiteRunRecord;
    suite?: RegressionSuiteSnapshot;
    error?: string;
  }>(response, 'Run regression CI suite');
  if (!response.ok || !data.run || !data.suite) {
    throw new Error(data.error || `Regression CI run failed (HTTP ${response.status}).`);
  }
  return { run: data.run, suite: data.suite };
}

/**
 * Fetches ready-to-commit GitHub Actions YAML, pytest harness, and JSON manifest.
 */
export async function fetchRegressionCiArtifacts(): Promise<RegressionCiArtifacts> {
  logFunctionCall('fetchRegressionCiArtifacts');
  const response = await fetch(APP_CONFIG.api.regressionCiBundle, {
    credentials: 'same-origin',
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    artifacts?: RegressionCiArtifacts;
    error?: string;
  }>(response, 'Fetch CI artifacts');
  if (!response.ok || !data.artifacts) {
    throw new Error(data.error || `Failed to fetch CI artifacts (HTTP ${response.status}).`);
  }
  return data.artifacts;
}

/**
 * Triggers a browser download from a binary ZIP API response and returns the
 * resolved archive filename.
 */
async function downloadZipFromEndpoint(
  url: string,
  payload: unknown,
  fallbackFilename: string,
  label: string,
): Promise<string> {
  const response = await fetch(url, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errText = await response.text();
    let message = `${label} download failed (HTTP ${response.status}).`;
    try {
      const parsed = JSON.parse(errText) as { error?: string };
      if (parsed.error) message = parsed.error;
    } catch {
      // Keep fallback message
    }
    throw new Error(message);
  }

  const disposition = response.headers.get('Content-Disposition') ?? '';
  const match = /filename="([^"]+)"/.exec(disposition);
  const filename = match?.[1] ?? fallbackFilename;

  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1500);
  return filename;
}

/**
 * Generates and fetches a preview of the Gymnasium + MuJoCo MJX RL package.
 */
export async function previewGymMjxPackage(
  config: GymMjxExportConfig,
): Promise<GymMjxPackageResponse> {
  logFunctionCall('previewGymMjxPackage', { ...config });
  const response = await fetch(APP_CONFIG.api.trainingGymPreview, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    package?: GymMjxPackageResponse;
    error?: string;
  }>(response, 'Gym/MJX package preview');
  if (!response.ok || !data.package) {
    throw new Error(data.error || `Failed to generate Gym/MJX package (HTTP ${response.status}).`);
  }
  return data.package;
}

/**
 * Downloads the complete Gymnasium + MuJoCo MJX RL training package as `.zip`.
 */
export async function downloadGymMjxPackageZip(
  config: GymMjxExportConfig,
): Promise<string> {
  logFunctionCall('downloadGymMjxPackageZip', { ...config });
  return downloadZipFromEndpoint(
    APP_CONFIG.api.trainingGymDownload,
    config,
    'lettherebesims_gym_mjx.zip',
    'Gym/MJX ZIP',
  );
}

/**
 * Generates synthetic domain-randomized LeRobot v2.1 episodes and returns the
 * quality report, per-feature normalization stats, and key metadata previews.
 */
export async function previewLeRobotDataset(
  config: LeRobotDatasetConfig,
): Promise<LeRobotDatasetPreviewResponse> {
  logFunctionCall('previewLeRobotDataset', { ...config });
  const response = await fetch(APP_CONFIG.api.trainingLerobotPreview, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  const data = await parseApiJsonResponse<{
    ok?: boolean;
    dataset?: LeRobotDatasetPreviewResponse;
    error?: string;
  }>(response, 'LeRobot dataset generation');
  if (!response.ok || !data.dataset) {
    throw new Error(data.error || `Failed to generate LeRobot dataset (HTTP ${response.status}).`);
  }
  return data.dataset;
}

/**
 * Downloads the complete Synthetic LeRobot v2.1 Episode Dataset as `.zip`.
 */
export async function downloadLeRobotDatasetZip(
  config: LeRobotDatasetConfig,
): Promise<string> {
  logFunctionCall('downloadLeRobotDatasetZip', { ...config });
  return downloadZipFromEndpoint(
    APP_CONFIG.api.trainingLerobotDownload,
    config,
    'lettherebesims_lerobot_dataset.zip',
    'LeRobot Dataset ZIP',
  );
}


