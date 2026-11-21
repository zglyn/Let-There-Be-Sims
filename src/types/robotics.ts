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
  | 'hand_demo_retarget'
  | 'scene_map_only';

export type ExecutionOutcome = 'failure' | 'success' | 'hand_demo' | 'no_robot';

export type VideoSlot = 'primary' | 'secondary';

export interface MobileSensorCapture {
  /** Normalized [0..1, 0..1] viewfinder coordinates where the judge tapped the target object. */
  tapTarget: { x: number; y: number } | null;
  /** Gravity vector in phone/camera frame (m/s^2) from DeviceOrientation/DeviceMotion. */
  gravityVector: { x: number; y: number; z: number } | null;
  /** Raw DeviceOrientation Euler angles in degrees plus derived camera elevation tilt. */
  orientationDeg: {
    alpha: number;
    beta: number;
    gamma: number;
    cameraTiltDeg: number;
  } | null;
  /** True when the recorded mobile clip includes a live microphone audio track. */
  hasAudioTrack: boolean;
  /** Actual recorded clip duration in seconds (default 6s judge hand demo). */
  clipDurationSec: number;
  /** Capture intent selected on the mobile viewfinder. */
  captureMode: 'hand_demo' | 'robot_footage' | 'scene_scan';
  capturedAt: string;
}

export interface HandDemoRetargetSpec {
  detected: boolean;
  spokenInstructionTranscript: string;
  retargetedRobot: string;
  obstacleAvoidedName: string;
  pickupPos: Vector3Tuple;
  apexWaypointPos: Vector3Tuple;
  placePos: Vector3Tuple;
  retargetNotes: string;
}

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
  handDemoRetarget?: HandDemoRetargetSpec | null;
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
  taskCompleted?: boolean;
  placementErrorM?: number;
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
  mobileSensors: MobileSensorCapture | null;
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
  | {
      type: 'video_uploaded';
      slot: VideoSlot;
      video: UploadedVideoMeta;
      sessionStatus: SessionStatus;
      mobileSensors?: MobileSensorCapture | null;
    }
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

export type RegressionRunMode =
  | 'candidate_policy'
  | 'suite_verified'
  | 'suite_baseline';

export interface RegressionCaseRunResult {
  caseId: string;
  caseName: string;
  robotModel: RobotModelId;
  passedEnvs: number;
  totalEnvs: number;
  status: 'pass' | 'fail';
  worstContactForceN: number;
  worstSlipVelocityMs: number;
  minLiftHeightM: number;
  tasksCompletedEnvs: number;
  failureReasons: string[];
  durationMs: number;
}

export interface RegressionTestCase {
  id: string;
  name: string;
  createdAt: string;
  sourceSessionId: string;
  taskGoal: string;
  robotModel: RobotModelId;
  executionOutcome: ExecutionOutcome;
  failureSummary: string;
  tags: string[];
  sceneSpec: SceneSpec;
  baselineControllerParams: ControllerParams;
  baselinePolicyCode: string;
  verifiedControllerParams: ControllerParams;
  verifiedPolicyCode: string;
  lastRun: RegressionCaseRunResult | null;
}

export interface RegressionSuiteRunRecord {
  runId: string;
  executedAt: string;
  mode: RegressionRunMode;
  policyLabel: string;
  totalCases: number;
  passedCases: number;
  failedCases: number;
  totalEnvsPassed: number;
  totalEnvsEvaluated: number;
  overallStatus: 'pass' | 'fail';
  ciExitCode: 0 | 1;
  durationMs: number;
  caseResults: RegressionCaseRunResult[];
}

export interface RegressionCiArtifacts {
  githubWorkflowYaml: string;
  pytestScriptPy: string;
  manifestJson: string;
  caseCount: number;
  generatedAt: string;
}

export interface RegressionSuiteSnapshot {
  cases: RegressionTestCase[];
  runs: RegressionSuiteRunRecord[];
}

export type GymRlAlgorithm = 'ppo' | 'sac' | 'mjx_ppo';
export type GymRewardPreset = 'dense_pick_place' | 'force_safe_precision' | 'sparse_goal';
export type GymRandomizationLevel = 'off' | 'standard' | 'aggressive';

export interface GymMjxExportConfig {
  sessionId?: string;
  regressionCaseId?: string;
  algorithm: GymRlAlgorithm;
  rewardPreset: GymRewardPreset;
  randomizationLevel: GymRandomizationLevel;
  numEnvs: number;
  totalTimesteps: number;
}

export interface TrainingPackageFile {
  path: string;
  label: string;
  language: 'python' | 'xml' | 'json' | 'bash' | 'markdown' | 'text';
  content: string;
  sizeBytes: number;
}

export interface GymMjxPackageResponse {
  packageName: string;
  sourceLabel: string;
  robotModel: RobotModelId;
  taskGoal: string;
  goalPad: { x: number; z: number; radiusM: number; label: string };
  config: {
    algorithm: GymRlAlgorithm;
    rewardPreset: GymRewardPreset;
    randomizationLevel: GymRandomizationLevel;
    numEnvs: number;
    totalTimesteps: number;
    controlHz: number;
    maxEpisodeSteps: number;
  };
  files: TrainingPackageFile[];
  generatedAt: string;
}

export type LeRobotPolicyTarget = 'act' | 'diffusion' | 'smolvla';
export type LeRobotRandomizationSpread = 'low' | 'medium' | 'high';

export interface LeRobotDatasetConfig {
  sessionId?: string;
  regressionCaseId?: string;
  episodeCount: number;
  fps: number;
  policyTarget: LeRobotPolicyTarget;
  randomizationSpread: LeRobotRandomizationSpread;
  includeFailedEpisodes: boolean;
}

export interface LeRobotEpisodeSummary {
  episodeIndex: number;
  lengthFrames: number;
  durationSec: number;
  passed: boolean;
  friction: number;
  massKg: number;
  xyOffset: [number, number];
  peakContactForceN: number;
  maxSlipVelocityMs: number;
  placementErrorM: number;
  totalReward: number;
}

export interface LeRobotFeatureStat {
  featureName: string;
  shape: number[];
  min: number[];
  max: number[];
  mean: number[];
  std: number[];
}

export interface LeRobotSampleFrame {
  frameIndex: number;
  timestamp: number;
  state: number[];
  action: number[];
  contactForceN: number;
  reward: number;
  done: boolean;
}

export interface LeRobotDatasetPreviewResponse {
  datasetName: string;
  repoIdSuggestion: string;
  sourceLabel: string;
  robotModel: RobotModelId;
  taskGoal: string;
  codebaseVersion: string;
  fps: number;
  policyTarget: LeRobotPolicyTarget;
  randomizationSpread: LeRobotRandomizationSpread;
  attemptedEpisodes: number;
  exportedEpisodes: number;
  filteredOutEpisodes: number;
  totalFrames: number;
  successRatePct: number;
  meanPeakForceN: number;
  meanPlacementErrorMm: number;
  episodes: LeRobotEpisodeSummary[];
  featureStats: LeRobotFeatureStat[];
  sampleFrames: LeRobotSampleFrame[];
  previewFiles: TrainingPackageFile[];
  generatedAt: string;
}


