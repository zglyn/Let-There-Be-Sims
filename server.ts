/**
 * Express Full-Stack Server & Vite Middleware Entry Point
 *
 * Serves the React frontend (via Vite dev middleware in development or static
 * `dist/` in production) alongside the LetThereBeSims backend API:
 * - `POST /api/sessions`: Create a QR-pairable compiler session.
 * - `GET /api/sessions/:id`: Fetch session snapshot.
 * - `GET /api/sessions/:id/events`: Live Server-Sent Events stream.
 * - `POST /api/sessions/:id/upload`: Validated multipart video upload from
 *   smartphone `/capture` or desktop intake, triggering SSE updates.
 * - `GET /api/sessions/:id/media`: Byte-range streaming for uploaded footage.
 * - `POST /api/sessions/:id/run`: Launches the Gemini 4 closed-loop pipeline.
 * - `POST /api/sessions/:id/webhook`: Saves SSRF-validated Cloud MuJoCo webhook.
 * - `POST /api/webhook/test`: Tests external Cloud MuJoCo webhook reachability.
 * - `GET /api/sessions/:id/bundle`: Polls background Long Decode bundle status.
 *
 * Use cases:
 * - Hosting both the desktop command center (`/`) and mobile camera capture
 *   (`/capture?session=<id>`) on port 3000 with server-side Gemini API access.
 */

import 'dotenv/config';
import crypto from 'crypto';
import express from 'express';
import fs from 'fs';
import multer from 'multer';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { APP_CONFIG, type RobotModelId } from './src/config/appConfig';
import {
  appendUploadChunk,
  finalizeChunkedUpload,
  initChunkedUpload,
  isAllowedMediaMime,
  resolveUploadedMimeType,
} from './src/server/chunkedUploadManager';
import {
  cancelFailureToPolicyPipeline,
  refreshLongDecodeBundle,
  runFailureToPolicyPipeline,
} from './src/server/pipelineOrchestrator';
import {
  broadcastSessionEvent,
  createSession,
  ensureUploadRoot,
  getOrRestoreSessionRecord,
  getSessionRecord,
  resolveConfinedUploadPath,
  subscribeToSessionEvents,
} from './src/server/sessionStore';
import {
  deleteRegressionCase,
  generateRegressionCiArtifacts,
  getRegressionCaseById,
  getRegressionSuiteSnapshot,
  runRegressionSuiteCi,
  saveSessionAsRegressionCase,
} from './src/server/regressionSuiteStore';
import {
  buildGymMjxTrainingPackage,
  buildGymMjxZipArchive,
  type GymMjxSourceInput,
} from './src/server/gymMjxExporter';
import {
  buildLeRobotDatasetPreview,
  buildLeRobotDatasetZip,
  type LeRobotGeneratorInput,
} from './src/server/lerobotDatasetGenerator';
import {
  testExternalWebhook,
  validateExternalWebhookUrl,
} from './src/server/webhookValidator';
import type {
  ControllerParams,
  GymRandomizationLevel,
  GymRewardPreset,
  GymRlAlgorithm,
  LeRobotPolicyTarget,
  LeRobotRandomizationSpread,
  MobileSensorCapture,
  RegressionRunMode,
  SceneSpec,
  UploadedVideoMeta,
  VideoSlot,
} from './src/types/robotics';
import { logFunctionCall } from './src/utils/logger';

/**
 * Configures Multer disk storage inside the confined temporary upload root with
 * server-generated random filenames and strict MIME/size filtering.
 */
function createVideoUploadMiddleware() {
  logFunctionCall('createVideoUploadMiddleware');
  const uploadRoot = ensureUploadRoot();

  const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadRoot),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase().replace(/[^.a-z0-9]/g, '') || '.mp4';
      const safeName = `clip_${Date.now()}_${crypto.randomBytes(6).toString('hex')}${ext}`;
      cb(null, safeName);
    },
  });

  return multer({
    storage,
    limits: {
      fileSize: APP_CONFIG.upload.maxFileSizeBytes,
      files: 1,
    },
    fileFilter: (_req, file, cb) => {
      const resolvedMime = resolveUploadedMimeType(file.mimetype, file.originalname);
      if (isAllowedMediaMime(resolvedMime)) {
        cb(null, true);
      } else {
        cb(
          new Error(
            `Unsupported file type '${file.mimetype || path.extname(file.originalname)}'. Upload a video (MP4/WebM/MOV) or frame image.`,
          ),
        );
      }
    },
  });
}

/**
 * Boots the Express server, registers all API routes, mounts Vite middleware
 * (or production static assets), and binds to port 3000.
 */
async function startServer(): Promise<void> {
  logFunctionCall('startServer', { port: APP_CONFIG.serverPort, nodeEnv: process.env.NODE_ENV ?? 'development' });
  const app = express();
  const upload = createVideoUploadMiddleware();

  app.use(express.json({ limit: '2mb' }));

  // 1. Create Session
  app.post('/api/sessions', (req, res) => {
    logFunctionCall('POST /api/sessions', { body: req.body });
    const { taskGoal, robotModel } = (req.body ?? {}) as {
      taskGoal?: string;
      robotModel?: RobotModelId;
    };
    const record = createSession(taskGoal, robotModel);
    res.status(201).json({ session: record.state });
  });

  // 2. Get Session Snapshot
  app.get('/api/sessions/:id', (req, res) => {
    logFunctionCall('GET /api/sessions/:id', { id: req.params.id });
    const record = getSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found. Create a new session on the desktop dashboard.' });
      return;
    }
    res.json({ session: record.state });
  });

  // 3. Server-Sent Events Stream
  app.get('/api/sessions/:id/events', (req, res) => {
    logFunctionCall('GET /api/sessions/:id/events', { id: req.params.id });
    const record = getOrRestoreSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }
    subscribeToSessionEvents(record, res);
  });

  // 4a. Chunked Binary Upload — Step 1: Initialize Upload Handle
  app.post('/api/sessions/:id/upload/init', (req, res) => {
    logFunctionCall('POST /api/sessions/:id/upload/init', { id: req.params.id });
    const record = getOrRestoreSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Invalid session ID. Refresh to create a fresh session.' });
      return;
    }

    try {
      const body = (req.body ?? {}) as {
        filename?: string;
        mimeType?: string;
        sizeBytes?: number;
        source?: string;
        slot?: string;
        taskGoal?: string;
      };
      const slot: VideoSlot = body.slot === 'secondary' ? 'secondary' : 'primary';
      const source: UploadedVideoMeta['source'] =
        body.source === 'mobile_qr' || body.source === 'desktop_webcam'
          ? body.source
          : 'desktop_upload';

      if (typeof body.taskGoal === 'string' && body.taskGoal.trim()) {
        record.state.taskGoal = body.taskGoal.trim().slice(0, 400);
      }

      const init = initChunkedUpload({
        sessionId: record.state.id,
        slot,
        filename: String(body.filename || 'capture.mp4'),
        mimeType: typeof body.mimeType === 'string' ? body.mimeType : undefined,
        sizeBytes: Number(body.sizeBytes ?? 0),
        source,
      });

      res.json({ ok: true, uploadId: init.uploadId, mimeType: init.mimeType, slot: init.slot });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Unable to initialize upload.',
      });
    }
  });

  // 4b. Chunked Binary Upload — Step 2: Append Raw Binary Chunk
  app.post(
    '/api/sessions/:id/upload/chunk',
    express.raw({ type: '*/*', limit: '4mb' }),
    (req, res) => {
      const uploadId = String(req.query.uploadId ?? '');
      const chunkIndex = Number(req.query.chunkIndex ?? 0);
      logFunctionCall('POST /api/sessions/:id/upload/chunk', {
        id: req.params.id,
        uploadId,
        chunkIndex,
      });

      const record = getOrRestoreSessionRecord(req.params.id);
      if (!record) {
        res.status(404).json({ error: 'Session not found.' });
        return;
      }

      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty binary chunk received.' });
        return;
      }

      try {
        const result = appendUploadChunk({
          sessionId: record.state.id,
          uploadId,
          chunkBuffer: req.body,
          chunkIndex,
        });
        res.json({ ok: true, receivedBytes: result.receivedBytes });
      } catch (err) {
        res.status(400).json({
          error: err instanceof Error ? err.message : 'Failed to append upload chunk.',
        });
      }
    },
  );

  /**
   * Validates and clamps optional smartphone tap-target, DeviceOrientation
   * gravity vector, and audio-track metadata submitted with mobile captures.
   */
  const sanitizeMobileSensors = (raw: unknown): MobileSensorCapture | null => {
    if (!raw || typeof raw !== 'object') return null;
    const obj = raw as Record<string, unknown>;
    const clampNum = (v: unknown, min: number, max: number, fallback: number): number =>
      typeof v === 'number' && Number.isFinite(v) ? Math.max(min, Math.min(max, v)) : fallback;

    const rawTap = obj.tapTarget as Record<string, unknown> | null | undefined;
    const tapTarget =
      rawTap && typeof rawTap === 'object' && typeof rawTap.x === 'number' && typeof rawTap.y === 'number'
        ? {
            x: Number(clampNum(rawTap.x, 0, 1, 0.5).toFixed(4)),
            y: Number(clampNum(rawTap.y, 0, 1, 0.5).toFixed(4)),
          }
        : null;

    const rawGrav = obj.gravityVector as Record<string, unknown> | null | undefined;
    const gravityVector =
      rawGrav &&
      typeof rawGrav === 'object' &&
      typeof rawGrav.x === 'number' &&
      typeof rawGrav.y === 'number' &&
      typeof rawGrav.z === 'number'
        ? {
            x: Number(clampNum(rawGrav.x, -25, 25, 0).toFixed(3)),
            y: Number(clampNum(rawGrav.y, -25, 25, -6.9).toFixed(3)),
            z: Number(clampNum(rawGrav.z, -25, 25, -6.9).toFixed(3)),
          }
        : null;

    const rawOri = obj.orientationDeg as Record<string, unknown> | null | undefined;
    const orientationDeg =
      rawOri && typeof rawOri === 'object'
        ? {
            alpha: Number(clampNum(rawOri.alpha, -360, 360, 0).toFixed(1)),
            beta: Number(clampNum(rawOri.beta, -180, 180, 45).toFixed(1)),
            gamma: Number(clampNum(rawOri.gamma, -90, 90, 0).toFixed(1)),
            cameraTiltDeg: Number(clampNum(rawOri.cameraTiltDeg, -90, 90, 45).toFixed(1)),
          }
        : null;

    const rawMode = String(obj.captureMode ?? 'hand_demo');
    const captureMode: MobileSensorCapture['captureMode'] =
      rawMode === 'robot_footage' || rawMode === 'scene_scan' ? rawMode : 'hand_demo';

    return {
      tapTarget,
      gravityVector,
      orientationDeg,
      hasAudioTrack: Boolean(obj.hasAudioTrack),
      clipDurationSec: Number(
        clampNum(obj.clipDurationSec, 1, 120, APP_CONFIG.mobileHandDemo.defaultClipDurationSec).toFixed(1),
      ),
      captureMode,
      capturedAt:
        typeof obj.capturedAt === 'string' && obj.capturedAt.trim()
          ? obj.capturedAt
          : new Date().toISOString(),
    };
  };

  // 4c. Chunked Binary Upload — Step 3: Finalize & Broadcast SSE Event
  app.post('/api/sessions/:id/upload/complete', (req, res) => {
    logFunctionCall('POST /api/sessions/:id/upload/complete', { id: req.params.id });
    const record = getOrRestoreSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }

    try {
      const body = (req.body ?? {}) as {
        uploadId?: string;
        autoStart?: boolean;
        mobileSensors?: unknown;
      };
      const { filePath, videoMeta, slot } = finalizeChunkedUpload({
        sessionId: record.state.id,
        uploadId: String(body.uploadId ?? ''),
      });

      const mobileSensors = sanitizeMobileSensors(body.mobileSensors);
      if (mobileSensors) {
        record.state.mobileSensors = mobileSensors;
        if (mobileSensors.captureMode === 'hand_demo') {
          record.state.robotModel = 'so_arm101';
          record.state.rolloutDurationSec = Math.max(
            APP_CONFIG.simulation.minRolloutDurationSec,
            Math.round(mobileSensors.clipDurationSec),
          );
        }
      }

      if (slot === 'secondary') {
        record.secondaryVideoFilePath = filePath;
        record.state.secondaryVideo = videoMeta;
      } else {
        record.videoFilePath = filePath;
        record.state.video = videoMeta;
      }

      if (record.state.video && record.state.status === 'awaiting_video') {
        record.state.status = 'video_ready';
      } else if (record.state.video && record.state.status !== 'running') {
        record.state.status = 'video_ready';
      }
      record.state.error = null;

      broadcastSessionEvent(record, {
        type: 'video_uploaded',
        slot,
        video: videoMeta,
        sessionStatus: record.state.status,
        mobileSensors: record.state.mobileSensors,
      });

      if (Boolean(body.autoStart) && record.state.video && !record.isPipelineRunning) {
        void runFailureToPolicyPipeline(record);
      }

      res.json({ ok: true, slot, video: videoMeta, session: record.state });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Failed to finalize video upload.',
      });
    }
  });

  // 4d. Legacy Multipart Upload Endpoint (retained for direct CLI/curl uploads)
  app.post('/api/sessions/:id/upload', (req, res) => {
    logFunctionCall('POST /api/sessions/:id/upload', { id: req.params.id });
    const record = getOrRestoreSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found. Re-scan the active QR code.' });
      return;
    }

    upload.single('video')(req, res, (uploadErr) => {
      if (uploadErr) {
        res.status(400).json({ error: uploadErr.message || 'Video upload failed.' });
        return;
      }
      if (!req.file) {
        res.status(400).json({ error: 'No video file received in form field "video".' });
        return;
      }

      const confinedPath = resolveConfinedUploadPath(req.file.filename);
      const slot: VideoSlot = String(req.body?.slot ?? 'primary') === 'secondary' ? 'secondary' : 'primary';
      const sourceRaw = String(req.body?.source ?? 'desktop_upload');
      const source: UploadedVideoMeta['source'] =
        sourceRaw === 'mobile_qr' || sourceRaw === 'desktop_webcam' ? sourceRaw : 'desktop_upload';

      if (typeof req.body?.taskGoal === 'string' && req.body.taskGoal.trim()) {
        record.state.taskGoal = req.body.taskGoal.trim().slice(0, 400);
      }

      const baseMediaUrl = APP_CONFIG.api.mediaFile(record.state.id, slot);
      const sep = baseMediaUrl.includes('?') ? '&' : '?';

      const videoMeta: UploadedVideoMeta = {
        originalName: req.file.originalname.slice(0, 120),
        mimeType: resolveUploadedMimeType(req.file.mimetype, req.file.originalname) || 'video/mp4',
        sizeBytes: req.file.size,
        uploadedAt: new Date().toISOString(),
        source,
        mediaUrl: `${baseMediaUrl}${sep}v=${Date.now()}`,
      };

      if (slot === 'secondary') {
        record.secondaryVideoFilePath = confinedPath;
        record.state.secondaryVideo = videoMeta;
      } else {
        record.videoFilePath = confinedPath;
        record.state.video = videoMeta;
      }
      if (record.state.video && record.state.status !== 'running') {
        record.state.status = 'video_ready';
      }
      record.state.error = null;

      broadcastSessionEvent(record, {
        type: 'video_uploaded',
        slot,
        video: videoMeta,
        sessionStatus: record.state.status,
      });

      // Optionally auto-start pipeline when uploaded from paired smartphone
      const autoStart = String(req.body?.autoStart ?? 'false') === 'true';
      if (autoStart && record.state.video && !record.isPipelineRunning) {
        void runFailureToPolicyPipeline(record);
      }

      res.json({ ok: true, slot, video: videoMeta, session: record.state });
    });
  });

  // 5. Stream Uploaded Media (Primary or Secondary Slot) with HTTP Byte-Range Support
  app.get('/api/sessions/:id/media', (req, res) => {
    const slot: VideoSlot = String(req.query.slot ?? 'primary') === 'secondary' ? 'secondary' : 'primary';
    logFunctionCall('GET /api/sessions/:id/media', { id: req.params.id, slot });
    const record = getSessionRecord(req.params.id);
    const targetPath = slot === 'secondary' ? record?.secondaryVideoFilePath : record?.videoFilePath;
    const targetMeta = slot === 'secondary' ? record?.state.secondaryVideo : record?.state.video;
    if (!record || !targetPath || !targetMeta) {
      res.status(404).json({ error: `No uploaded ${slot} footage found for this session.` });
      return;
    }

    const confinedPath = resolveConfinedUploadPath(path.basename(targetPath));
    if (!fs.existsSync(confinedPath)) {
      res.status(404).json({ error: 'Uploaded footage file is no longer available.' });
      return;
    }

    const stat = fs.statSync(confinedPath);
    if (stat.size === 0) {
      res.status(404).json({ error: 'Uploaded footage file is empty.' });
      return;
    }

    const range = req.headers.range;
    const contentType = targetMeta.mimeType || 'video/mp4';

    if (range && range.startsWith('bytes=')) {
      const rawSpec = range.replace(/^bytes=/, '').split(',')[0].trim();
      const [startStr, endStr] = rawSpec.split('-');
      let start: number;
      let end: number;

      if (startStr === '') {
        // Suffix byte-range (e.g. bytes=-500)
        const suffixLen = parseInt(endStr, 10);
        start = Number.isNaN(suffixLen) ? 0 : Math.max(0, stat.size - suffixLen);
        end = stat.size - 1;
      } else {
        start = parseInt(startStr, 10);
        const requestedEnd = endStr ? parseInt(endStr, 10) : stat.size - 1;
        // Clamp end byte to stat.size - 1 per RFC 9110 (browsers often request large chunk windows)
        end = Number.isNaN(requestedEnd) ? stat.size - 1 : Math.min(requestedEnd, stat.size - 1);
      }

      if (Number.isNaN(start) || start < 0 || start >= stat.size || start > end) {
        res.setHeader('Content-Range', `bytes */${stat.size}`);
        res.status(416).end();
        return;
      }

      const chunkSize = end - start + 1;
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': contentType,
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(confinedPath, { start, end }).pipe(res);
    } else {
      res.writeHead(200, {
        'Accept-Ranges': 'bytes',
        'Content-Length': stat.size,
        'Content-Type': contentType,
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(confinedPath).pipe(res);
    }
  });

  // 5b. Remove Optional Secondary (or Primary) Uploaded Video Slot
  app.delete('/api/sessions/:id/media', (req, res) => {
    const slot: VideoSlot = String(req.query.slot ?? 'secondary') === 'primary' ? 'primary' : 'secondary';
    logFunctionCall('DELETE /api/sessions/:id/media', { id: req.params.id, slot });
    const record = getOrRestoreSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }

    if (slot === 'secondary') {
      record.secondaryVideoFilePath = null;
      record.state.secondaryVideo = null;
    } else {
      record.videoFilePath = null;
      record.state.video = null;
      if (record.state.status === 'video_ready') {
        record.state.status = 'awaiting_video';
      }
    }

    broadcastSessionEvent(record, {
      type: 'video_removed',
      slot,
      sessionStatus: record.state.status,
    });

    res.json({ ok: true, slot, session: record.state });
  });

  // 6. Launch Gemini 4 Failure-to-Policy Pipeline
  app.post('/api/sessions/:id/run', (req, res) => {
    logFunctionCall('POST /api/sessions/:id/run', { id: req.params.id, body: req.body });
    const record = getSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }
    if (!record.videoFilePath || !record.state.video) {
      res.status(400).json({ error: 'Upload or record a tabletop video before starting compilation.' });
      return;
    }
    if (record.isPipelineRunning) {
      res.status(409).json({ error: 'Compilation pipeline is already active for this session.' });
      return;
    }

    const { taskGoal, robotModel, forceSimulateRobot, rolloutDurationSec } = (req.body ?? {}) as {
      taskGoal?: string;
      robotModel?: RobotModelId;
      forceSimulateRobot?: boolean;
      rolloutDurationSec?: number;
    };
    if (typeof taskGoal === 'string' && taskGoal.trim()) {
      record.state.taskGoal = taskGoal.trim().slice(0, 400);
    }
    if (robotModel && robotModel in APP_CONFIG.robots) {
      record.state.robotModel = robotModel;
    }
    if (typeof rolloutDurationSec === 'number' && Number.isFinite(rolloutDurationSec)) {
      record.state.rolloutDurationSec = Math.max(
        APP_CONFIG.simulation.minRolloutDurationSec,
        Math.min(APP_CONFIG.simulation.maxRolloutDurationSec, Math.round(rolloutDurationSec)),
      );
    }

    void runFailureToPolicyPipeline(record, {
      forceSimulateRobot: Boolean(forceSimulateRobot),
    });
    res.status(202).json({ ok: true, status: 'running' });
  });

  // 6b. Cancel Active Gemini 4 Pipeline Run
  app.post('/api/sessions/:id/cancel', (req, res) => {
    logFunctionCall('POST /api/sessions/:id/cancel', { id: req.params.id });
    const record = getSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }
    const cancelled = cancelFailureToPolicyPipeline(record);
    res.json({ ok: true, cancelled, session: record.state });
  });

  // 7. Configure External Cloud MuJoCo Webhook
  app.post('/api/sessions/:id/webhook', async (req, res) => {
    logFunctionCall('POST /api/sessions/:id/webhook', {
      id: req.params.id,
      enabled: req.body?.enabled,
      url: req.body?.url,
    });
    const record = getSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }

    const enabled = Boolean(req.body?.enabled);
    const url = typeof req.body?.url === 'string' ? req.body.url.trim() : '';
    const authToken = typeof req.body?.authToken === 'string' ? req.body.authToken.trim() : null;

    if (enabled) {
      const validation = await validateExternalWebhookUrl(url);
      if (!validation.valid || !validation.normalizedUrl) {
        res.status(400).json({ error: validation.error ?? 'Invalid webhook URL.' });
        return;
      }
      record.state.webhook = {
        ...record.state.webhook,
        enabled: true,
        url: validation.normalizedUrl,
        hasAuthToken: Boolean(authToken ?? record.webhookAuthToken),
      };
      if (authToken !== null) {
        record.webhookAuthToken = authToken || null;
      }
    } else {
      record.state.webhook = {
        ...record.state.webhook,
        enabled: false,
        url,
        hasAuthToken: Boolean(record.webhookAuthToken),
      };
    }

    broadcastSessionEvent(record, { type: 'session_snapshot', session: record.state });
    res.json({ ok: true, webhook: record.state.webhook });
  });

  // 8. Test External Cloud MuJoCo Webhook Reachability
  app.post('/api/webhook/test', async (req, res) => {
    logFunctionCall('POST /api/webhook/test', { url: req.body?.url });
    const url = typeof req.body?.url === 'string' ? req.body.url : '';
    const authToken = typeof req.body?.authToken === 'string' ? req.body.authToken : null;
    const result = await testExternalWebhook(url, authToken);
    res.status(result.ok ? 200 : 400).json(result);
  });

  // 9. Poll Long Decode Bundle Status
  app.get('/api/sessions/:id/bundle', async (req, res) => {
    logFunctionCall('GET /api/sessions/:id/bundle', { id: req.params.id });
    const record = getSessionRecord(req.params.id);
    if (!record) {
      res.status(404).json({ error: 'Session not found.' });
      return;
    }
    try {
      const bundle = await refreshLongDecodeBundle(record);
      res.json({ bundle });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh Long Decode bundle.';
      res.status(502).json({ error: message });
    }
  });

  // 10. Failure-to-Regression-Suite CI/CD: List Saved Cases & CI Build History
  app.get('/api/regression/cases', (_req, res) => {
    logFunctionCall('GET /api/regression/cases');
    const snapshot = getRegressionSuiteSnapshot();
    res.json(snapshot);
  });

  // 10b. Promote Active Session Failure/Scene into Permanent Regression Test Case
  app.post('/api/regression/cases', (req, res) => {
    logFunctionCall('POST /api/regression/cases', { sessionId: req.body?.sessionId });
    const sessionId = typeof req.body?.sessionId === 'string' ? req.body.sessionId.trim() : '';
    const customName = typeof req.body?.name === 'string' ? req.body.name : undefined;
    const customTags = Array.isArray(req.body?.tags)
      ? req.body.tags.filter((t: unknown): t is string => typeof t === 'string')
      : undefined;

    const record = getSessionRecord(sessionId);
    if (!record) {
      res.status(404).json({ error: 'Active session not found.' });
      return;
    }
    try {
      const testCase = saveSessionAsRegressionCase({
        session: record.state,
        customName,
        customTags,
      });
      res.status(201).json({ ok: true, testCase, suite: getRegressionSuiteSnapshot() });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Unable to save regression test case.',
      });
    }
  });

  // 10c. Delete a Saved Regression Test Case
  app.delete('/api/regression/cases/:caseId', (req, res) => {
    logFunctionCall('DELETE /api/regression/cases/:caseId', { caseId: req.params.caseId });
    const removed = deleteRegressionCase(req.params.caseId);
    if (!removed) {
      res.status(404).json({ error: 'Regression test case not found.' });
      return;
    }
    res.json({ ok: true, suite: getRegressionSuiteSnapshot() });
  });

  // 10d. Run Full Regression Suite CI Gate (returns ciExitCode 0 = PASS, 1 = FAIL)
  app.post('/api/regression/run', (req, res) => {
    logFunctionCall('POST /api/regression/run', {
      mode: req.body?.mode,
      sessionId: req.body?.sessionId,
      policyLabel: req.body?.policyLabel,
    });
    const rawMode = String(req.body?.mode ?? 'suite_verified');
    const mode: RegressionRunMode =
      rawMode === 'candidate_policy' || rawMode === 'suite_baseline'
        ? rawMode
        : 'suite_verified';
    const policyLabel = typeof req.body?.policyLabel === 'string' ? req.body.policyLabel : undefined;

    let candidateControllerParams: ControllerParams | null = null;
    let candidateRobotModel: RobotModelId | undefined;

    if (mode === 'candidate_policy') {
      if (typeof req.body?.sessionId === 'string' && req.body.sessionId.trim()) {
        const record = getSessionRecord(req.body.sessionId.trim());
        if (!record) {
          res.status(404).json({ error: 'Candidate session not found.' });
          return;
        }
        const latestIter = record.state.iterations[record.state.iterations.length - 1];
        candidateControllerParams =
          latestIter?.controllerParams ?? record.state.sceneSpec?.controllerParams ?? null;
        candidateRobotModel = record.state.robotModel;
      } else if (req.body?.controllerParams && typeof req.body.controllerParams === 'object') {
        candidateControllerParams = req.body.controllerParams as ControllerParams;
        if (typeof req.body?.robotModel === 'string' && req.body.robotModel in APP_CONFIG.robots) {
          candidateRobotModel = req.body.robotModel as RobotModelId;
        }
      }
      if (!candidateControllerParams) {
        res.status(400).json({
          error: 'No compiled policy found in the active session to test against the regression suite.',
        });
        return;
      }
    }

    try {
      const run = runRegressionSuiteCi({
        mode,
        policyLabel,
        candidateControllerParams,
        candidateRobotModel,
      });
      res.json({ ok: true, run, suite: getRegressionSuiteSnapshot() });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Failed to execute regression suite CI run.',
      });
    }
  });

  // 10e. Export Ready-to-Commit GitHub Actions YAML & pytest Suite Bundle
  app.get('/api/regression/ci-bundle', (_req, res) => {
    logFunctionCall('GET /api/regression/ci-bundle');
    const artifacts = generateRegressionCiArtifacts();
    res.json({ ok: true, artifacts });
  });

  /**
   * Resolves a calibrated tabletop scene, verified controller, task goal, and
   * robot model from either an active session or a saved Regression CI case.
   */
  const resolveTrainingExportSource = (body: Record<string, unknown>): {
    sourceLabel: string;
    taskGoal: string;
    robotModel: RobotModelId;
    sceneSpec: SceneSpec;
    controllerParams: ControllerParams;
  } | { error: string; status: number } => {
    const regressionCaseId =
      typeof body.regressionCaseId === 'string' ? body.regressionCaseId.trim() : '';
    const sessionId = typeof body.sessionId === 'string' ? body.sessionId.trim() : '';

    if (regressionCaseId) {
      const regCase = getRegressionCaseById(regressionCaseId);
      if (!regCase) {
        return { error: 'Selected Regression CI case not found.', status: 404 };
      }
      return {
        sourceLabel: regCase.name,
        taskGoal: regCase.taskGoal,
        robotModel: regCase.robotModel,
        sceneSpec: regCase.sceneSpec,
        controllerParams: regCase.verifiedControllerParams,
      };
    }

    if (sessionId) {
      const record = getSessionRecord(sessionId);
      if (!record) {
        return { error: 'Active compiler session not found.', status: 404 };
      }
      if (!record.state.sceneSpec) {
        return {
          error:
            'Compile a scene first (or select a saved Regression CI case) before generating training exports.',
          status: 400,
        };
      }
      const latestIter = record.state.iterations[record.state.iterations.length - 1];
      return {
        sourceLabel: `${record.state.sceneSpec.targetObject.name} · Active Session`,
        taskGoal: record.state.taskGoal,
        robotModel: record.state.robotModel,
        sceneSpec: record.state.sceneSpec,
        controllerParams: latestIter?.controllerParams ?? record.state.sceneSpec.controllerParams,
      };
    }

    return {
      error: 'Provide either sessionId or regressionCaseId to build training exports.',
      status: 400,
    };
  };

  /** Parses and clamps Gymnasium + MJX export options from request body. */
  const parseGymMjxInput = (
    body: Record<string, unknown>,
  ): GymMjxSourceInput | { error: string; status: number } => {
    const source = resolveTrainingExportSource(body);
    if ('error' in source) return source;

    const rawAlgo = String(body.algorithm ?? APP_CONFIG.trainingExport.gymMjx.defaultAlgo);
    const algorithm: GymRlAlgorithm =
      rawAlgo === 'sac' || rawAlgo === 'mjx_ppo' ? rawAlgo : 'ppo';

    const rawReward = String(
      body.rewardPreset ?? APP_CONFIG.trainingExport.gymMjx.defaultRewardPreset,
    );
    const rewardPreset: GymRewardPreset =
      rawReward === 'force_safe_precision' || rawReward === 'sparse_goal'
        ? rawReward
        : 'dense_pick_place';

    const rawDr = String(
      body.randomizationLevel ?? APP_CONFIG.trainingExport.gymMjx.defaultRandomizationLevel,
    );
    const randomizationLevel: GymRandomizationLevel =
      rawDr === 'off' || rawDr === 'aggressive' ? rawDr : 'standard';

    const rawEnvs = Number(body.numEnvs ?? APP_CONFIG.trainingExport.gymMjx.defaultNumEnvs);
    const numEnvs = Number.isFinite(rawEnvs)
      ? Math.max(1, Math.min(8192, Math.round(rawEnvs)))
      : APP_CONFIG.trainingExport.gymMjx.defaultNumEnvs;

    const rawSteps = Number(
      body.totalTimesteps ?? APP_CONFIG.trainingExport.gymMjx.defaultTotalTimesteps,
    );
    const totalTimesteps = Number.isFinite(rawSteps)
      ? Math.max(10_000, Math.min(50_000_000, Math.round(rawSteps)))
      : APP_CONFIG.trainingExport.gymMjx.defaultTotalTimesteps;

    return {
      ...source,
      algorithm,
      rewardPreset,
      randomizationLevel,
      numEnvs,
      totalTimesteps,
    };
  };

  /** Parses and clamps Synthetic LeRobot v2.1 dataset options from request body. */
  const parseLeRobotInput = (
    body: Record<string, unknown>,
  ): LeRobotGeneratorInput | { error: string; status: number } => {
    const source = resolveTrainingExportSource(body);
    if ('error' in source) return source;

    const rawEpisodes = Number(
      body.episodeCount ?? APP_CONFIG.trainingExport.lerobot.defaultEpisodeCount,
    );
    const episodeCount = Number.isFinite(rawEpisodes)
      ? Math.max(
          1,
          Math.min(APP_CONFIG.trainingExport.lerobot.maxEpisodeCount, Math.round(rawEpisodes)),
        )
      : APP_CONFIG.trainingExport.lerobot.defaultEpisodeCount;

    const rawFps = Number(body.fps ?? APP_CONFIG.trainingExport.lerobot.defaultFps);
    const fps = APP_CONFIG.trainingExport.lerobot.fpsPresets.includes(rawFps)
      ? rawFps
      : APP_CONFIG.trainingExport.lerobot.defaultFps;

    const rawPolicy = String(
      body.policyTarget ?? APP_CONFIG.trainingExport.lerobot.defaultPolicyTarget,
    );
    const policyTarget: LeRobotPolicyTarget =
      rawPolicy === 'diffusion' || rawPolicy === 'smolvla' ? rawPolicy : 'act';

    const rawSpread = String(
      body.randomizationSpread ?? APP_CONFIG.trainingExport.lerobot.defaultRandomizationSpread,
    );
    const randomizationSpread: LeRobotRandomizationSpread =
      rawSpread === 'low' || rawSpread === 'high' ? rawSpread : 'medium';

    return {
      ...source,
      episodeCount,
      fps,
      policyTarget,
      randomizationSpread,
      includeFailedEpisodes: Boolean(body.includeFailedEpisodes),
    };
  };

  // 11a. Preview Gymnasium + MuJoCo MJX RL Training Package
  app.post('/api/training/gym-mjx/preview', (req, res) => {
    logFunctionCall('POST /api/training/gym-mjx/preview', {
      sessionId: req.body?.sessionId ?? null,
      regressionCaseId: req.body?.regressionCaseId ?? null,
      algorithm: req.body?.algorithm,
    });
    const parsed = parseGymMjxInput((req.body ?? {}) as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(parsed.status).json({ error: parsed.error });
      return;
    }
    const pkg = buildGymMjxTrainingPackage(parsed);
    res.json({ ok: true, package: pkg });
  });

  // 11b. Download Gymnasium + MuJoCo MJX RL Training Package (.zip)
  app.post('/api/training/gym-mjx/download', (req, res) => {
    logFunctionCall('POST /api/training/gym-mjx/download', {
      sessionId: req.body?.sessionId ?? null,
      regressionCaseId: req.body?.regressionCaseId ?? null,
      algorithm: req.body?.algorithm,
    });
    const parsed = parseGymMjxInput((req.body ?? {}) as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(parsed.status).json({ error: parsed.error });
      return;
    }
    const { filename, buffer } = buildGymMjxZipArchive(parsed);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', String(buffer.length));
    res.send(buffer);
  });

  // 12a. Generate & Preview Synthetic LeRobot v2.1 Episode Dataset
  app.post('/api/training/lerobot/preview', (req, res) => {
    logFunctionCall('POST /api/training/lerobot/preview', {
      sessionId: req.body?.sessionId ?? null,
      regressionCaseId: req.body?.regressionCaseId ?? null,
      episodeCount: req.body?.episodeCount,
      fps: req.body?.fps,
      policyTarget: req.body?.policyTarget,
    });
    const parsed = parseLeRobotInput((req.body ?? {}) as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(parsed.status).json({ error: parsed.error });
      return;
    }
    const dataset = buildLeRobotDatasetPreview(parsed);
    res.json({ ok: true, dataset });
  });

  // 12b. Download Synthetic LeRobot v2.1 Episode Dataset (.zip)
  app.post('/api/training/lerobot/download', (req, res) => {
    logFunctionCall('POST /api/training/lerobot/download', {
      sessionId: req.body?.sessionId ?? null,
      regressionCaseId: req.body?.regressionCaseId ?? null,
      episodeCount: req.body?.episodeCount,
      fps: req.body?.fps,
    });
    const parsed = parseLeRobotInput((req.body ?? {}) as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(parsed.status).json({ error: parsed.error });
      return;
    }
    const { filename, buffer } = buildLeRobotDatasetZip(parsed);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', String(buffer.length));
    res.send(buffer);
  });

  // Guard unmatched /api/* endpoints so they always return JSON 404 (never HTML 200)
  app.use('/api', (req, res) => {
    res.status(404).json({ error: `API route not found: ${req.method} ${req.originalUrl}` });
  });

  // Mount Vite Dev Middleware or Static Production Build
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(APP_CONFIG.serverPort, '0.0.0.0', () => {
    console.info(
      `[INFO][${new Date().toISOString()}][SERVER] LetThereBeSims listening on http://0.0.0.0:${APP_CONFIG.serverPort}`,
    );
  });
}

void startServer();
