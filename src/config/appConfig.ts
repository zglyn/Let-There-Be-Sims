/**
 * Centralized Application Configuration Module
 *
 * This module serves as the single source of truth for all configurable values
 * across both the Express backend and React frontend of LetThereBeSims.
 * Centralizing model identifiers, thinking levels, physics verification gates,
 * swarm counts, upload limits, API endpoints, and 3D camera presets ensures
 * deterministic consistency across the multi-model Gemini 4 pipeline and makes
 * rapid calibration changes effortless during live robotics demonstrations.
 *
 * Use cases:
 * - Switching or auditing Gemini 4 model IDs and thinking_level escalations.
 * - Tuning sim-to-real success criteria (lift height target, peak force ceiling).
 * - Adjusting 20-environment domain randomization envelopes and iteration caps.
 * - Configuring 3D stage camera presets and robot kinematic link dimensions.
 */

export const APP_CONFIG = {
  appName: 'LetThereBeSims',
  serverPort: 3000,

  /** Gemini 4 multi-model pipeline roles and thinking_level escalations. */
  models: {
    spatialArchitect: {
      id: 'gemini-4-carbon' as const,
      label: 'Gemini 4 Carbon',
      role: 'Spatial Real-to-Sim Architect',
      thinkingLevel: 'medium' as const,
    },
    telemetryAggregator: {
      id: 'gemini-4-hydrogen' as const,
      label: 'Gemini 4 Hydrogen',
      role: 'Cloud Swarm Telemetry Triage',
      thinkingLevel: 'low' as const,
    },
    policyPatcher: {
      id: 'gemini-4-neon' as const,
      label: 'Gemini 4 Neon',
      role: 'Adversarial Sim-to-Real Debugger',
      thinkingLevel: 'xhigh' as const,
    },
    bundleCompiler: {
      id: 'gemini-4-neon' as const,
      label: 'Gemini 4 Neon (Long Decode)',
      role: 'Deployment Bundle Compiler',
      thinkingLevel: 'high' as const,
    },
  },

  /** Closed-loop physics verification gates and swarm configuration. */
  simulation: {
    swarmSize: 20,
    maxRepairIterations: 4,
    rolloutSteps: 1500,
    controlHz: 500,
    replayFrameCount: 60,
    maxReplayFrameCount: 180,
    defaultRolloutDurationSec: 15,
    minRolloutDurationSec: 3,
    maxRolloutDurationSec: 120,
    durationPresetsSec: [10, 30, 60] as readonly number[],
    targetLiftZMeters: 0.12,
    maxSafeContactForceN: 45.0,
    maxSafeSlipVelocityMs: 0.08,
    videoExport: {
      fps: 30,
      videoBitsPerSecond: 4_500_000,
      defaultDurationSec: 60,
      presetsSec: [15, 30, 60] as readonly number[],
    },
    defaultTaskGoal:
      'Grasp the ceramic mug cleanly from the cluttered tabletop and lift it 15 cm without colliding with adjacent obstacles.',
    defaultRobotModel: 'so_arm101' as const,
    liveSandbox: {
      gravityMs2: 9.81,
      defaultFriction: 0.44,
      defaultMassKg: 0.28,
      defaultClearanceM: 0.11,
      defaultGripForceN: 24.0,
      nudgeImpulseMs: 0.42,
      tableBoundsX: [0.1, 0.38] as readonly [number, number],
      tableBoundsZ: [-0.24, 0.24] as readonly [number, number],
      jointDampLambda: 11.0,
      compactHeightPx: 480,
      expandedHeightPx: 660,
    },
  },

  /**
   * Authoritative stage progress weights and expected durations used by the
   * backend orchestrator and client asymptotic progress estimator.
   */
  pipelineProgress: {
    mediaPrepStartPct: 3,
    mediaPrepTargetPct: 8,
    mediaPrepExpectedSec: 6,
    carbonStartPct: 8,
    carbonTargetPct: 35,
    carbonExpectedSec: 32,
    swarmStartPct: 38,
    swarmTargetPct: 44,
    swarmExpectedSec: 4,
    hydrogenStartPct: 44,
    hydrogenTargetPct: 55,
    hydrogenExpectedSec: 10,
    neonStartPct: 56,
    neonTargetPct: 78,
    neonExpectedSec: 34,
    reverifySpanPerIterPct: 8,
    longDecodeStartPct: 88,
    longDecodeTargetPct: 98,
    longDecodeExpectedSec: 8,
    clientTickIntervalMs: 350,
    typicalTotalDurationLabel: '45–120s typical',
  },

  /** Robot kinematics definitions used by the 3D Three.js stage and evaluator. */
  robots: {
    so_arm101: {
      id: 'so_arm101' as const,
      name: 'SO-ARM101 (6-DOF)',
      manufacturer: 'LeRobot Open-Source',
      dof: 6,
      maxTorqueNm: 4.8,
      reachMeters: 0.44,
      baseHeightM: 0.085,
      upperArmM: 0.165,
      forearmM: 0.155,
      wristM: 0.08,
    },
    franka_panda: {
      id: 'franka_panda' as const,
      name: 'Franka Emika Panda (7-DOF)',
      manufacturer: 'MuJoCo Menagerie',
      dof: 7,
      maxTorqueNm: 87.0,
      reachMeters: 0.855,
      baseHeightM: 0.11,
      upperArmM: 0.21,
      forearmM: 0.19,
      wristM: 0.095,
    },
  },

  /** Upload constraints, video proxy transcoding, and security boundaries. */
  upload: {
    maxFileSizeBytes: 60 * 1024 * 1024, // 60 MB
    chunkSizeBytes: 1.5 * 1024 * 1024, // 1.5 MB per fetch chunk (bypasses proxy body limits)
    inlineBase64LimitBytes: 2 * 1024 * 1024, // Prefer Gemini Files API above 2 MB
    analysisProxy: {
      enabled: true,
      minBytesToTranscode: 2.5 * 1024 * 1024, // Transcode videos > 2.5 MB to 1 FPS / 480p proxy
      targetFps: 1,
      maxWidthPx: 480,
      maxDurationSec: 180,
      crf: 28,
      transcodeTimeoutMs: 45_000,
    },
    allowedMimeTypes: [
      'video/mp4',
      'video/webm',
      'video/quicktime',
      'video/x-m4v',
      'video/3gpp',
      'image/jpeg',
      'image/png',
    ],
    webhookTimeoutMs: 45_000,
  },

  /** Mobile QR Judge Hand-Demo capture (tap-to-select + 6s clip + gravity + audio). */
  mobileHandDemo: {
    defaultClipDurationSec: 6,
    enableAudioCapture: true,
    proxyAudioBitrateKbps: 32,
    shortClipProxyFps: 2,
    shortClipMaxBytes: 15 * 1024 * 1024,
  },

  /** Hard per-stage timeouts and background polling intervals for Gemini 4 API calls. */
  timeouts: {
    createHandshakeMs: 90_000,
    stagePollIntervalMs: 3_000,
    carbonStageMs: 180_000, // 3 minutes hard ceiling for Stage 1 (even on 2-view long clips)
    hydrogenStageMs: 90_000,
    neonStageMs: 180_000,
    longDecodeDispatchMs: 60_000,
  },

  /** Failure-to-Regression-Suite CI/CD persistence and verification thresholds. */
  regressionCi: {
    storageDirName: 'lettherebesims_regression_suite',
    maxRunHistory: 30,
    requiredPassRate: 1.0, // 100% of randomized envs across all saved failure cases
    autoPromoteFailures: true,
  },

  /** Exportable RL (Gymnasium + MuJoCo MJX) and LeRobot synthetic dataset configuration. */
  trainingExport: {
    gymMjx: {
      defaultAlgo: 'ppo' as const,
      algos: ['ppo', 'sac', 'mjx_ppo'] as readonly string[],
      defaultRewardPreset: 'dense_pick_place' as const,
      rewardPresets: ['dense_pick_place', 'force_safe_precision', 'sparse_goal'] as readonly string[],
      defaultRandomizationLevel: 'standard' as const,
      randomizationLevels: ['off', 'standard', 'aggressive'] as readonly string[],
      defaultNumEnvs: 64,
      numEnvsPresets: [16, 64, 1024, 4096] as readonly number[],
      defaultTotalTimesteps: 1_000_000,
      timestepPresets: [250_000, 1_000_000, 5_000_000] as readonly number[],
      controlHz: 50,
      maxEpisodeSteps: 250,
    },
    lerobot: {
      codebaseVersion: 'v2.1' as const,
      defaultEpisodeCount: 25,
      episodeCountPresets: [10, 25, 50, 100] as readonly number[],
      maxEpisodeCount: 150,
      defaultFps: 20,
      fpsPresets: [10, 20, 30] as readonly number[],
      defaultPolicyTarget: 'act' as const,
      policyTargets: ['act', 'diffusion', 'smolvla'] as readonly string[],
      defaultRandomizationSpread: 'medium' as const,
      randomizationSpreads: ['low', 'medium', 'high'] as readonly string[],
      defaultEpisodeDurationSec: 6.5,
    },
  },

  /** REST & SSE endpoint builders. */
  api: {
    createSession: '/api/sessions',
    getSession: (id: string) => `/api/sessions/${encodeURIComponent(id)}`,
    uploadVideo: (id: string) => `/api/sessions/${encodeURIComponent(id)}/upload`,
    uploadInit: (id: string) => `/api/sessions/${encodeURIComponent(id)}/upload/init`,
    uploadChunk: (id: string, uploadId: string, chunkIndex: number) =>
      `/api/sessions/${encodeURIComponent(id)}/upload/chunk?uploadId=${encodeURIComponent(uploadId)}&chunkIndex=${chunkIndex}`,
    uploadComplete: (id: string) => `/api/sessions/${encodeURIComponent(id)}/upload/complete`,
    runPipeline: (id: string) => `/api/sessions/${encodeURIComponent(id)}/run`,
    cancelPipeline: (id: string) => `/api/sessions/${encodeURIComponent(id)}/cancel`,
    eventsStream: (id: string) => `/api/sessions/${encodeURIComponent(id)}/events`,
    configureWebhook: (id: string) => `/api/sessions/${encodeURIComponent(id)}/webhook`,
    testWebhook: '/api/webhook/test',
    bundleStatus: (id: string) => `/api/sessions/${encodeURIComponent(id)}/bundle`,
    mediaFile: (id: string, slot: 'primary' | 'secondary' = 'primary') =>
      slot === 'secondary'
        ? `/api/sessions/${encodeURIComponent(id)}/media?slot=secondary`
        : `/api/sessions/${encodeURIComponent(id)}/media`,
    regressionCases: '/api/regression/cases',
    regressionCase: (caseId: string) => `/api/regression/cases/${encodeURIComponent(caseId)}`,
    regressionRun: '/api/regression/run',
    regressionCiBundle: '/api/regression/ci-bundle',
    trainingGymPreview: '/api/training/gym-mjx/preview',
    trainingGymDownload: '/api/training/gym-mjx/download',
    trainingLerobotPreview: '/api/training/lerobot/preview',
    trainingLerobotDownload: '/api/training/lerobot/download',
  },

  /** Interactive 3D Three.js camera presets (coordinates in meters, Y-up). */
  cameraPresets: {
    perspective: {
      label: 'Perspective',
      position: [0.58, 0.44, 0.62] as [number, number, number],
      target: [0.06, 0.12, 0.0] as [number, number, number],
    },
    top: {
      label: 'Top Calibration',
      position: [0.06, 0.86, 0.02] as [number, number, number],
      target: [0.06, 0.04, 0.0] as [number, number, number],
    },
    gripper: {
      label: 'Gripper Close-Up',
      position: [0.28, 0.24, 0.28] as [number, number, number],
      target: [0.16, 0.09, 0.02] as [number, number, number],
    },
    side: {
      label: 'Side Elevation',
      position: [0.05, 0.24, 0.72] as [number, number, number],
      target: [0.05, 0.12, 0.0] as [number, number, number],
    },
  },

  /** Multi-viewport stage layouts for robotic arm inspection. */
  stageViewModes: {
    dual_cam: {
      id: 'dual_cam' as const,
      label: 'Dual Cam',
      description: 'Workcell 3D + Wrist Eye-in-Hand / Ortho view',
    },
    dual_arms: {
      id: 'dual_arms' as const,
      label: 'Dual Arms',
      description: 'SO-ARM101 (6-DOF) & Franka Panda (7-DOF) side-by-side',
    },
    single: {
      id: 'single' as const,
      label: 'Single',
      description: 'Full-width single 3D viewport',
    },
  },

  /** Secondary camera options rendered in the right pane during Dual Cam mode. */
  secondaryCameraModes: {
    wrist: { id: 'wrist' as const, label: 'Wrist Cam', fov: 58 },
    top: { id: 'top' as const, label: 'Top-Down', fov: 40 },
    side: { id: 'side' as const, label: 'Side', fov: 40 },
    gripper: { id: 'gripper' as const, label: 'Gripper', fov: 40 },
  },
} as const;

export type RobotModelId = keyof typeof APP_CONFIG.robots;
export type CameraPresetId = keyof typeof APP_CONFIG.cameraPresets;
export type StageViewModeId = keyof typeof APP_CONFIG.stageViewModes;
export type SecondaryCameraModeId = keyof typeof APP_CONFIG.secondaryCameraModes;
