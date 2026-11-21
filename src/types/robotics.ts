/**
 * Robotics Domain & Pipeline Type Definitions
 *
 * Defines the strict TypeScript contracts shared across the Express API,
 * Gemini 4 Interactions orchestrator, physics swarm evaluator, external Cloud
 * MuJoCo webhook bridge, Three.js 3D viewport, and React command center UI.
 *
 * Use cases:
 * - End-to-end type safety for SSE events streamed from server to browser.
 * - Structured JSON schemas emitted by Gemini 4 Carbon, Hydrogen, and Neon.
 * - Deterministic 60-frame kinematics playback and contact telemetry curves.
 */

import type { RobotModelId } from '../config/appConfig';

export type SessionStatus =
  | 'awaiting_video'
  | 'video_ready'
  | 'running'
  | 'completed'
  | 'error';

export type PipelineStageKey =
  | 'carbon_reconstruct'
  | 'swarm_hydrogen_triage'
  | 'neon_policy_patch'
  | 'neon_long_decode';

export type StageStatus = 'idle' | 'active' | 'completed' | 'skipped' | 'error';

export type SceneMode =
  | 'robot_failure_repair'
  | 'robot_success_verification'
  | 'scene_map_only';

export type ExecutionOutcome = 'failure' | 'success' | 'no_robot';

export type VideoSlot = 'primary' | 'secondary';

export interface Vector3Tuple {
  x: number;
  y: number;
  z: number;
}

export interface SpatialZoneSpec {
  label: string;
  kind: 'clear_zone' | 'hazard_zone' | 'robot_mount';
  centerX: number;
  centerZ: number;
  radiusM: number;
  notes: string;
}

export interface SceneObjectSpec {
  name: string;
  shape: 'cylinder' | 'box' | 'mug' | 'bottle' | 'sphere';
  position: Vector3Tuple;
  dimensions: Vector3Tuple;
  colorHex: string;
  massKg: number;
  friction: number;
  materialLabel: string;
}

export interface DomainRandomizationParam {
  id: number;
  friction: number;
  mass: number;
  xyOffset: [number, number];
}

/**
 * Structured control parameters extracted alongside the Python policy so both
 * the built-in physics evaluator and the 3D Three.js kinematics solver mirror
 * every single patch made by Gemini 4 Neon.
 */
export interface ControllerParams {
  approachHeightM: number;
  preGraspOffsetX: number;
  preGraspOffsetY: number;
  descentSpeedMs: number;
  kpStiffness: number;
  kdDamping: number;
  gripForceN: number;
  gripRampSteps: number;
  clearanceArcM: number;
  liftHeightM: number;
}

export interface SceneSpec {
  hasRobotInVideo: boolean;
  executionOutcome: ExecutionOutcome;
  outcomeSummary: string;
  sceneMode: SceneMode;
  detectedRobotDescription: string;
  failureAnalysis: string;
  calibrationNotes: string;
  tableDimensions: Vector3Tuple;
  recommendedRobotBase: { x: number; z: number };
  targetObject: SceneObjectSpec;
  obstacles: SceneObjectSpec[];
  spatialZones: SpatialZoneSpec[];
  mjcfXml: string;
  initialControllerPython: string;
  controllerParams: ControllerParams;
  randomizations: DomainRandomizationParam[];
}

export interface TrajectoryFrame {
  timeSec: number;
  jointAngles: [number, number, number, number, number, number];
  gripperOpen: number; // 0 = closed, 1 = open
  endEffectorPos: Vector3Tuple;
  objectPos: Vector3Tuple;
  objectTiltRad: number;
  contactForceN: number;
  slipVelocityMs: number;
}

export interface RolloutResult {
  envId: number;
  success: boolean;
  failureReason: string | null;
  finalObjectZ: number;
  peakContactForceN: number;
  maxSlipVelocityMs: number;
  durationSec?: number;
  params: DomainRandomizationParam;
  trajectory: TrajectoryFrame[];
  previewVideoUrl?: string | null;
}

export interface InteractionTraceEntry {
  id: string;
  previousInteractionId: string | null;
  stage: PipelineStageKey;
  model: string;
  thinkingLevel: string;
  iteration?: number;
  title: string;
  thoughtSummary: string;
  outputExcerpt: string;
  durationMs: number;
  timestamp: string;
}

export interface IterationRecord {
  iteration: number;
  passed: number;
  total: number;
  executionSource: 'builtin_physics' | 'external_webhook';
  rollouts: RolloutResult[];
  telemetrySummary: string;
  rootCauseFix: string | null;
  policyCode: string;
  previousPolicyCode: string | null;
  controllerParams: ControllerParams;
  hydrogenInteractionId: string;
  neonInteractionId: string | null;
}

export interface DeploymentBundle {
  interactionId: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  standaloneControllerPy: string;
  regressionSuiteXml: string;
  auditReportMarkdown: string;
  rawText?: string;
  updatedAt: string;
  error?: string;
}

export interface WebhookConfig {
  enabled: boolean;
  url: string;
  hasAuthToken: boolean;
  lastTestedAt?: string;
  lastTestStatus?: 'ok' | 'error';
  lastTestMessage?: string;
}

export interface UploadedVideoMeta {
  originalName: string;
  mimeType: string;
  sizeBytes: number;
  uploadedAt: string;
  source: 'mobile_qr' | 'desktop_upload' | 'desktop_webcam';
  mediaUrl: string;
}

export interface PipelineStageState {
  key: PipelineStageKey;
  label: string;
  model: string;
  thinkingLevel: string;
  status: StageStatus;
  detail: string;
  interactionId: string | null;
  previousInteractionId: string | null;
}

export interface PipelineProgressState {
  percent: number;
  targetPercent: number;
  expectedStageDurationSec: number;
  stageLabel: string;
  stageUpdatedAt: string;
  startedAt: string;
}

export interface CompilerSessionState {
  id: string;
  createdAt: string;
  status: SessionStatus;
  taskGoal: string;
  robotModel: RobotModelId;
  rolloutDurationSec: number;
  video: UploadedVideoMeta | null;
  secondaryVideo: UploadedVideoMeta | null;
  webhook: WebhookConfig;
  stages: PipelineStageState[];
  progress: PipelineProgressState | null;
  sceneSpec: SceneSpec | null;
  iterations: IterationRecord[];
  interactionTrace: InteractionTraceEntry[];
  bundle: DeploymentBundle | null;
  error: string | null;
}

export type ServerStreamEvent =
  | { type: 'session_snapshot'; session: CompilerSessionState }
  | { type: 'video_uploaded'; slot: VideoSlot; video: UploadedVideoMeta; sessionStatus: SessionStatus }
  | { type: 'video_removed'; slot: VideoSlot; sessionStatus: SessionStatus }
  | {
      type: 'stage_update';
      stages: PipelineStageState[];
      status: SessionStatus;
      progress: PipelineProgressState | null;
    }
  | { type: 'trace_appended'; entry: InteractionTraceEntry }
  | { type: 'scene_ready'; sceneSpec: SceneSpec }
  | { type: 'iteration_complete'; iteration: IterationRecord }
  | { type: 'bundle_update'; bundle: DeploymentBundle }
  | { type: 'pipeline_error'; message: string }
  | { type: 'pipeline_complete'; session: CompilerSessionState };
