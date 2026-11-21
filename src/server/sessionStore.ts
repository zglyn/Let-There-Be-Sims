/**
 * Session Registry & Server-Sent Events (SSE) Broadcaster
 *
 * Manages active Sim2Real Compiler sessions on the Express server, stores
 * uploaded video paths with strict directory confinement, tracks per-session
 * Gemini 4 interaction traces and rollout iterations, and broadcasts live
 * pipeline updates to all connected desktop and mobile SSE subscribers.
 *
 * Use cases:
 * - Creating a new paired session when the desktop command center loads.
 * - Delivering instant phone-upload notifications from `/capture?session=<id>`
 *   to the desktop viewport without page refreshes.
 * - Replaying full session state when a browser tab reconnects mid-pipeline.
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import type { Response } from 'express';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type {
  CompilerSessionState,
  PipelineStageState,
  ServerStreamEvent,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

export interface InternalSessionRecord {
  state: CompilerSessionState;
  videoFilePath: string | null;
  secondaryVideoFilePath: string | null;
  webhookAuthToken: string | null;
  subscribers: Set<Response>;
  isPipelineRunning: boolean;
  abortController: AbortController | null;
}

const UPLOAD_ROOT = path.resolve(os.tmpdir(), 'sim2real_compiler_uploads');
const sessions = new Map<string, InternalSessionRecord>();

/**
 * Ensures the temporary upload confinement directory exists on disk.
 */
export function ensureUploadRoot(): string {
  logFunctionCall('ensureUploadRoot', { uploadRoot: UPLOAD_ROOT });
  if (!fs.existsSync(UPLOAD_ROOT)) {
    fs.mkdirSync(UPLOAD_ROOT, { recursive: true });
  }
  return UPLOAD_ROOT;
}

/**
 * Resolves a filename inside the confined upload directory and rejects any
 * path traversal attempts outside the base root.
 */
export function resolveConfinedUploadPath(filename: string): string {
  logFunctionCall('resolveConfinedUploadPath', { filename });
  const baseDir = ensureUploadRoot();
  const safeBaseName = path.basename(filename).replace(/[^a-zA-Z0-9._-]/g, '_');
  const resolved = path.resolve(baseDir, safeBaseName);
  if (!resolved.startsWith(baseDir + path.sep)) {
    throw new Error('Resolved upload path escapes confined upload directory.');
  }
  return resolved;
}

/**
 * Builds the default idle 4-stage Gemini 4 pipeline descriptor array.
 */
export function createInitialStages(): PipelineStageState[] {
  logFunctionCall('createInitialStages');
  return [
    {
      key: 'carbon_reconstruct',
      label: '01. Spatial Real-to-Sim Reconstruction',
      model: APP_CONFIG.models.spatialArchitect.id,
      thinkingLevel: APP_CONFIG.models.spatialArchitect.thinkingLevel,
      status: 'idle',
      detail: 'Awaiting smartphone or desktop video footage for metric calibration.',
      interactionId: null,
      previousInteractionId: null,
    },
    {
      key: 'swarm_hydrogen_triage',
      label: '02. 20-Env Swarm & Telemetry Triage',
      model: APP_CONFIG.models.telemetryAggregator.id,
      thinkingLevel: APP_CONFIG.models.telemetryAggregator.thinkingLevel,
      status: 'idle',
      detail: 'Executes 20 domain-randomized rollouts and clusters failure modes.',
      interactionId: null,
      previousInteractionId: null,
    },
    {
      key: 'neon_policy_patch',
      label: '03. Adversarial Policy Repair Loop',
      model: APP_CONFIG.models.policyPatcher.id,
      thinkingLevel: APP_CONFIG.models.policyPatcher.thinkingLevel,
      status: 'idle',
      detail: 'Patches grasp approach, impedance gains, and clearance until 20/20 pass.',
      interactionId: null,
      previousInteractionId: null,
    },
    {
      key: 'neon_long_decode',
      label: '04. Long Decode Deployment Bundle',
      model: APP_CONFIG.models.bundleCompiler.id,
      thinkingLevel: APP_CONFIG.models.bundleCompiler.thinkingLevel,
      status: 'idle',
      detail: 'Compiles LeRobot controller, 20 MJCF scenes, and dynamics audit report.',
      interactionId: null,
      previousInteractionId: null,
    },
  ];
}

/**
 * Creates and registers a fresh compiler session with a cryptographically
 * random session identifier.
 */
export function createSession(
  taskGoal?: string,
  robotModel?: RobotModelId,
): InternalSessionRecord {
  logFunctionCall('createSession', { taskGoal, robotModel });
  const id = `s2r_${crypto.randomBytes(6).toString('hex')}`;
  const state: CompilerSessionState = {
    id,
    createdAt: new Date().toISOString(),
    status: 'awaiting_video',
    taskGoal: (taskGoal && taskGoal.trim()) || APP_CONFIG.simulation.defaultTaskGoal,
    robotModel: robotModel && robotModel in APP_CONFIG.robots ? robotModel : APP_CONFIG.simulation.defaultRobotModel,
    rolloutDurationSec: APP_CONFIG.simulation.defaultRolloutDurationSec,
    video: null,
    secondaryVideo: null,
    webhook: {
      enabled: false,
      url: '',
      hasAuthToken: false,
    },
    stages: createInitialStages(),
    progress: null,
    sceneSpec: null,
    iterations: [],
    interactionTrace: [],
    bundle: null,
    error: null,
  };

  const record: InternalSessionRecord = {
    state,
    videoFilePath: null,
    secondaryVideoFilePath: null,
    webhookAuthToken: null,
    subscribers: new Set<Response>(),
    isPipelineRunning: false,
    abortController: null,
  };

  sessions.set(id, record);
  return record;
}

/**
 * Retrieves an existing session record by ID, validating safe ID formatting.
 */
export function getSessionRecord(sessionId: string): InternalSessionRecord | null {
  logFunctionCall('getSessionRecord', { sessionId });
  if (!/^s2r_[a-f0-9]{12}$/.test(sessionId)) {
    return null;
  }
  return sessions.get(sessionId) ?? null;
}

/**
 * Retrieves an existing session record or transparently restores a valid-format
 * session ID (`s2r_<12-hex>`) if the in-memory store restarted while a browser
 * tab or paired smartphone QR code remained open.
 */
export function getOrRestoreSessionRecord(sessionId: string): InternalSessionRecord | null {
  logFunctionCall('getOrRestoreSessionRecord', { sessionId });
  if (!/^s2r_[a-f0-9]{12}$/.test(sessionId)) {
    return null;
  }
  const existing = sessions.get(sessionId);
  if (existing) return existing;

  // Enforce bounded session registry size
  if (sessions.size >= 200) {
    const oldestKey = sessions.keys().next().value;
    if (oldestKey) sessions.delete(oldestKey);
  }

  const state: CompilerSessionState = {
    id: sessionId,
    createdAt: new Date().toISOString(),
    status: 'awaiting_video',
    taskGoal: APP_CONFIG.simulation.defaultTaskGoal,
    robotModel: APP_CONFIG.simulation.defaultRobotModel,
    rolloutDurationSec: APP_CONFIG.simulation.defaultRolloutDurationSec,
    video: null,
    secondaryVideo: null,
    webhook: {
      enabled: false,
      url: '',
      hasAuthToken: false,
    },
    stages: createInitialStages(),
    progress: null,
    sceneSpec: null,
    iterations: [],
    interactionTrace: [],
    bundle: null,
    error: null,
  };

  const restored: InternalSessionRecord = {
    state,
    videoFilePath: null,
    secondaryVideoFilePath: null,
    webhookAuthToken: null,
    subscribers: new Set<Response>(),
    isPipelineRunning: false,
    abortController: null,
  };
  sessions.set(sessionId, restored);
  return restored;
}

/**
 * Registers an Express SSE response stream on a session and pushes the current
 * session snapshot immediately.
 */
export function subscribeToSessionEvents(record: InternalSessionRecord, res: Response): void {
  logFunctionCall('subscribeToSessionEvents', { sessionId: record.state.id });
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  record.subscribers.add(res);
  sendSseEvent(res, { type: 'session_snapshot', session: record.state });

  const heartbeat = setInterval(() => {
    try {
      res.write(': heartbeat\n\n');
    } catch {
      clearInterval(heartbeat);
      record.subscribers.delete(res);
    }
  }, 20_000);

  res.on('close', () => {
    clearInterval(heartbeat);
    record.subscribers.delete(res);
  });
}

/**
 * Writes a single typed SSE event payload to one connected client stream.
 */
function sendSseEvent(res: Response, event: ServerStreamEvent): void {
  res.write(`data: ${JSON.stringify(event)}\n\n`);
}

/**
 * Broadcasts a typed SSE event to all desktop/mobile clients subscribed to the
 * specified session.
 */
export function broadcastSessionEvent(
  record: InternalSessionRecord,
  event: ServerStreamEvent,
): void {
  logFunctionCall('broadcastSessionEvent', {
    sessionId: record.state.id,
    eventType: event.type,
    subscriberCount: record.subscribers.size,
  });
  for (const res of record.subscribers) {
    try {
      sendSseEvent(res, event);
    } catch {
      record.subscribers.delete(res);
    }
  }
}
