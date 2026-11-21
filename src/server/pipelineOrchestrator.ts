/**
 * Gemini 4 Closed-Loop Failure-to-Policy & Scene Map Pipeline Orchestrator
 *
 * Implements the adaptive 4-stage stateful multi-model orchestration:
 * 1. Gemini 4 Carbon (`thinking_level: "high"`): Analyzes the uploaded
 *    smartphone/desktop footage, detects whether a robotic arm or robot is
 *    present (`hasRobotInVideo`), calibrates 3D metric poses & friction,
 *    compiles simulation-ready MuJoCo MJCF XML, 20 domain randomizations,
 *    spatial map zones, and an initial baseline controller.
 *    - If NO robot is visible (`hasRobotInVideo === false`) and
 *      `forceSimulateRobot` is false, the pipeline completes cleanly in
 *      **Scene Map Mode (`scene_map_only`)**—producing the simulation-ready 3D
 *      scene, MuJoCo MJCF XML, and interactive 2D top-down spatial map while
 *      skipping policy repair until the user optionally clicks "Mount Robot &
 *      Simulate Policy".
 * 2. Cloud MuJoCo Swarm + Gemini 4 Hydrogen (`thinking_level: "low"`): Fans out
 *    all 20 randomized environments (via External Cloud Webhook or Built-In
 *    Physics Evaluator) and chains via `previous_interaction_id` to cluster
 *    contact force and slip anomalies.
 * 3. Gemini 4 Neon (`thinking_level: "xhigh"`): Chains from Hydrogen to patch
 *    `execute_policy(model, data, step)` and control parameters, looping with
 *    the 20-environment swarm until 20/20 (100%) pass.
 * 4. Gemini 4 Neon (`background: true`, `thinking_level: "high"`): Launches
 *    Long Decode compilation of the production LeRobot/MuJoCo controller,
 *    20-scene MJCF regression suite, and Sim-to-Real Contact Dynamics Audit.
 *
 * Use cases:
 * - End-to-end failure diagnosis and policy patching when footage shows a robot.
 * - Real-to-Sim workspace reconstruction & 2D spatial mapping when footage shows
 *   only a cluttered tabletop, desk, or room without any robot arm present.
 * - On-demand robot mounting onto an already-mapped scene reusing Carbon's
 *   cached `previous_interaction_id`.
 */

import { APP_CONFIG } from '../config/appConfig';
import type {
  ControllerParams,
  DeploymentBundle,
  DomainRandomizationParam,
  ExecutionOutcome,
  InteractionTraceEntry,
  IterationRecord,
  PipelineProgressState,
  PipelineStageKey,
  SceneMode,
  SceneSpec,
  SpatialZoneSpec,
  StageStatus,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import {
  createLoggedInteraction,
  getLoggedInteraction,
  parseStructuredJson,
  prepareMediaForInteractions,
} from './geminiClient';
import { evaluateBuiltInSwarm } from './physicsEvaluator';
import {
  broadcastSessionEvent,
  createInitialStages,
  type InternalSessionRecord,
} from './sessionStore';
import { executeExternalSwarmWebhook } from './webhookValidator';

/**
 * Updates the authoritative numeric progress floor, active stage ceiling,
 * expected stage duration, and descriptive label on the session.
 */
function setPipelineProgress(
  record: InternalSessionRecord,
  params: {
    percent: number;
    targetPercent: number;
    expectedStageDurationSec: number;
    stageLabel: string;
    startedAt?: string;
  },
): void {
  const nowIso = new Date().toISOString();
  const prevPercent = record.state.progress?.percent ?? 0;
  const monotonicPercent = params.startedAt
    ? params.percent
    : Math.max(prevPercent, params.percent);
  const clampedPercent = Math.max(0, Math.min(100, Math.round(monotonicPercent)));
  const clampedTarget = Math.max(clampedPercent, Math.min(100, Math.round(params.targetPercent)));

  const nextProgress: PipelineProgressState = {
    percent: clampedPercent,
    targetPercent: clampedTarget,
    expectedStageDurationSec: Math.max(1, params.expectedStageDurationSec),
    stageLabel: params.stageLabel,
    stageUpdatedAt: nowIso,
    startedAt: params.startedAt ?? record.state.progress?.startedAt ?? nowIso,
  };
  record.state.progress = nextProgress;
}

/**
 * Updates a single stage entry on the session and broadcasts the stage state
 * transition (including numeric progress) to all connected SSE clients.
 */
function updateStage(
  record: InternalSessionRecord,
  key: PipelineStageKey,
  patch: Partial<{
    status: StageStatus;
    detail: string;
    interactionId: string | null;
    previousInteractionId: string | null;
  }>,
): void {
  record.state.stages = record.state.stages.map((stage) =>
    stage.key === key ? { ...stage, ...patch } : stage,
  );
  broadcastSessionEvent(record, {
    type: 'stage_update',
    stages: record.state.stages,
    status: record.state.status,
    progress: record.state.progress,
  });
}

/**
 * Appends a verified Gemini 4 interaction turn to the session trace and emits
 * it over SSE for real-time proof-of-work rendering.
 */
function appendTrace(record: InternalSessionRecord, entry: InteractionTraceEntry): void {
  record.state.interactionTrace = [...record.state.interactionTrace, entry];
  broadcastSessionEvent(record, { type: 'trace_appended', entry });
}

/**
 * Normalizes and validates the 20 domain-randomized perturbation specs returned
 * by Gemini 4 Carbon so downstream MuJoCo and 3D renderers always receive 20
 * complete, well-bounded environments.
 */
function normalizeRandomizations(
  rawList:
    | Array<{ id?: number; friction?: number; mass?: number; xyOffset?: number[]; xy_offset?: number[] }>
    | undefined,
  baseFriction: number,
  baseMass: number,
): DomainRandomizationParam[] {
  const result: DomainRandomizationParam[] = [];
  const total = APP_CONFIG.simulation.swarmSize;

  for (let i = 0; i < total; i++) {
    const src = rawList?.[i];
    const rawOffset = src?.xyOffset ?? src?.xy_offset;
    const dx =
      Array.isArray(rawOffset) && typeof rawOffset[0] === 'number'
        ? Math.max(-0.045, Math.min(0.045, rawOffset[0]))
        : Number((((i % 5) - 2) * 0.011).toFixed(4));
    const dy =
      Array.isArray(rawOffset) && typeof rawOffset[1] === 'number'
        ? Math.max(-0.045, Math.min(0.045, rawOffset[1]))
        : Number((Math.floor(i / 5) - 1.5) * 0.012);

    const friction =
      typeof src?.friction === 'number' && Number.isFinite(src.friction)
        ? Math.max(0.15, Math.min(1.1, src.friction))
        : Number(Math.max(0.18, baseFriction + ((i % 4) - 1.5) * 0.12).toFixed(2));
    const mass =
      typeof src?.mass === 'number' && Number.isFinite(src.mass)
        ? Math.max(0.08, Math.min(1.5, src.mass))
        : Number(Math.max(0.1, baseMass + ((i % 5) - 2) * 0.045).toFixed(3));

    result.push({
      id: i + 1,
      friction,
      mass,
      xyOffset: [Number(dx.toFixed(4)), Number(dy.toFixed(4))],
    });
  }
  return result;
}

/**
 * Normalizes spatial map zones returned by Gemini 4 Carbon, synthesizing a
 * sensible recommended robot mount zone if none was explicitly provided.
 */
function normalizeSpatialZones(
  rawZones: Array<Partial<SpatialZoneSpec>> | undefined,
  recommendedBase: { x: number; z: number },
): SpatialZoneSpec[] {
  const validKinds = new Set<SpatialZoneSpec['kind']>([
    'clear_zone',
    'hazard_zone',
    'robot_mount',
  ]);
  const cleaned: SpatialZoneSpec[] = (rawZones ?? []).slice(0, 6).map((z, idx) => ({
    label: (z.label && z.label.trim()) || `Zone ${idx + 1}`,
    kind: validKinds.has(z.kind as SpatialZoneSpec['kind'])
      ? (z.kind as SpatialZoneSpec['kind'])
      : 'clear_zone',
    centerX: typeof z.centerX === 'number' && Number.isFinite(z.centerX) ? z.centerX : 0.15,
    centerZ: typeof z.centerZ === 'number' && Number.isFinite(z.centerZ) ? z.centerZ : 0,
    radiusM:
      typeof z.radiusM === 'number' && Number.isFinite(z.radiusM)
        ? Math.max(0.04, Math.min(0.24, z.radiusM))
        : 0.08,
    notes: z.notes ?? 'Calibrated spatial region',
  }));

  if (!cleaned.some((z) => z.kind === 'robot_mount')) {
    cleaned.unshift({
      label: 'Recommended Robot Mount',
      kind: 'robot_mount',
      centerX: recommendedBase.x,
      centerZ: recommendedBase.z,
      radiusM: 0.075,
      notes: 'Clear tabletop anchor position with unobstructed reach to target object.',
    });
  }
  return cleaned;
}

/**
 * Parses a structured or markdown-delimited Long Decode deployment bundle text
 * into downloadable standalone Python, MJCF regression XML, and Markdown audit.
 */
export function splitBundleSections(
  rawText: string,
  fallbackPolicy: string,
  fallbackXml: string,
): {
  standaloneControllerPy: string;
  regressionSuiteXml: string;
  auditReportMarkdown: string;
} {
  logFunctionCall('splitBundleSections', { rawLength: rawText.length });
  const pyMatch = rawText.match(/```python\s*([\s\S]*?)```/i);
  const xmlMatch = rawText.match(/```xml\s*([\s\S]*?)```/i);

  return {
    standaloneControllerPy: pyMatch?.[1]?.trim() || fallbackPolicy,
    regressionSuiteXml: xmlMatch?.[1]?.trim() || fallbackXml,
    auditReportMarkdown: rawText.trim(),
  };
}

/**
 * Executes the adaptive Gemini 4 pipeline for a session:
 * - Automatically detects whether a robot is present in the video.
 * - If no robot is detected (and `forceSimulateRobot` is false), produces a
 *   simulation-ready 3D scene + MuJoCo MJCF + 2D Spatial Map cleanly.
 * - If a robot is present, or if the user explicitly triggers
 *   `forceSimulateRobot: true` on an existing mapped scene, executes the full
 *   20-environment swarm, Hydrogen triage, Neon `xhigh` policy repair loop, and
 *   Long Decode deployment bundle compilation.
 */
export async function runFailureToPolicyPipeline(
  record: InternalSessionRecord,
  options: { forceSimulateRobot?: boolean } = {},
): Promise<void> {
  const forceSimulateRobot = Boolean(options.forceSimulateRobot);
  logFunctionCall('runFailureToPolicyPipeline', {
    sessionId: record.state.id,
    taskGoal: record.state.taskGoal,
    robotModel: record.state.robotModel,
    webhookEnabled: record.state.webhook.enabled,
    forceSimulateRobot,
  });

  if (record.isPipelineRunning) {
    throw new Error('Pipeline is already running for this session.');
  }
  if (!record.videoFilePath || !record.state.video) {
    throw new Error('Upload a smartphone or desktop video before compiling.');
  }

  record.isPipelineRunning = true;
  const abortController = new AbortController();
  record.abortController = abortController;
  record.state.status = 'running';
  record.state.error = null;

  const canReuseMappedScene = Boolean(
    forceSimulateRobot &&
      record.state.sceneSpec &&
      record.state.stages.find((s) => s.key === 'carbon_reconstruct')?.interactionId,
  );

  if (!canReuseMappedScene) {
    record.state.iterations = [];
    record.state.interactionTrace = [];
    record.state.sceneSpec = null;
    record.state.bundle = null;
    record.state.stages = createInitialStages();
  } else {
    record.state.iterations = [];
    record.state.bundle = null;
    record.state.stages = record.state.stages.map((stage) =>
      stage.key === 'carbon_reconstruct'
        ? stage
        : { ...stage, status: 'idle', interactionId: null },
    );
  }

  const startedAtIso = new Date().toISOString();
  const progCfg = APP_CONFIG.pipelineProgress;
  setPipelineProgress(record, {
    percent: canReuseMappedScene ? progCfg.carbonTargetPct : progCfg.mediaPrepStartPct,
    targetPercent: canReuseMappedScene ? progCfg.swarmTargetPct : progCfg.mediaPrepTargetPct,
    expectedStageDurationSec: canReuseMappedScene
      ? progCfg.swarmExpectedSec
      : progCfg.mediaPrepExpectedSec,
    stageLabel: canReuseMappedScene
      ? 'Stage 2/4 · Mounting Robot & Running 20-Env Swarm'
      : 'Stage 1/4 · Preparing Video Footage',
    startedAt: startedAtIso,
  });

  broadcastSessionEvent(record, {
    type: 'stage_update',
    stages: record.state.stages,
    status: record.state.status,
    progress: record.state.progress,
  });

  try {
    const robotSpec = APP_CONFIG.robots[record.state.robotModel];

    let sceneSpec: SceneSpec;
    let lastInteractionId: string;
    let currentPolicyCode: string;
    let currentControllerParams: ControllerParams;

    if (canReuseMappedScene && record.state.sceneSpec) {
      // Reuse already-compiled Carbon spatial map & cached interaction ID
      sceneSpec = {
        ...record.state.sceneSpec,
        sceneMode: 'robot_failure_repair',
      };
      record.state.sceneSpec = sceneSpec;
      lastInteractionId =
        record.state.stages.find((s) => s.key === 'carbon_reconstruct')?.interactionId ?? '';
      currentPolicyCode = sceneSpec.initialControllerPython;
      currentControllerParams = { ...sceneSpec.controllerParams };
      broadcastSessionEvent(record, { type: 'scene_ready', sceneSpec });
    } else {
      // -----------------------------------------------------------------------
      // STAGE 1: GEMINI 4 CARBON — Spatial Reconstruction & Robot Presence Check
      // -----------------------------------------------------------------------
      const hasSecondaryMedia = Boolean(
        record.secondaryVideoFilePath && record.state.secondaryVideo,
      );

      setPipelineProgress(record, {
        percent: progCfg.mediaPrepStartPct,
        targetPercent: progCfg.mediaPrepTargetPct,
        expectedStageDurationSec: progCfg.mediaPrepExpectedSec,
        stageLabel: hasSecondaryMedia
          ? 'Stage 1/4 · Preparing 2 Camera Views'
          : 'Stage 1/4 · Preparing Video Footage',
      });

      updateStage(record, 'carbon_reconstruct', {
        status: 'active',
        detail: hasSecondaryMedia
          ? 'Optimizing 2 camera views (1 FPS analysis proxy) and preparing Gemini 4 Carbon calibration...'
          : 'Optimizing video footage (1 FPS analysis proxy) and preparing Gemini 4 Carbon calibration...',
      });

      const primaryMediaPart = await prepareMediaForInteractions(
        record.videoFilePath,
        record.state.video.mimeType,
        abortController.signal,
      );
      const secondaryMediaPart =
        hasSecondaryMedia && record.secondaryVideoFilePath && record.state.secondaryVideo
          ? await prepareMediaForInteractions(
              record.secondaryVideoFilePath,
              record.state.secondaryVideo.mimeType,
              abortController.signal,
            )
          : null;

      updateStage(record, 'carbon_reconstruct', {
        status: 'active',
        detail: hasSecondaryMedia
          ? 'Fusing 2 camera views — detecting robot execution outcome (success vs. failure) and calibrating metric 3D scene with Gemini 4 Carbon...'
          : 'Inspecting footage for robot presence & execution outcome (success vs. failure) and calibrating metric 3D scene with Gemini 4 Carbon...',
      });

      setPipelineProgress(record, {
        percent: progCfg.carbonStartPct,
        targetPercent: progCfg.carbonTargetPct,
        expectedStageDurationSec: progCfg.carbonExpectedSec,
        stageLabel: hasSecondaryMedia
          ? 'Stage 1/4 · Gemini 4 Carbon (2-View Calibration)'
          : 'Stage 1/4 · Gemini 4 Carbon (Spatial Reconstruction)',
      });
      broadcastSessionEvent(record, {
        type: 'stage_update',
        stages: record.state.stages,
        status: record.state.status,
        progress: record.state.progress,
      });

      const carbonResponseFormat = {
        type: 'text',
        mime_type: 'application/json',
        schema: {
          type: 'object',
          properties: {
            hasRobotInVideo: { type: 'boolean' },
            executionOutcome: { type: 'string' },
            outcomeSummary: { type: 'string' },
            detectedRobotDescription: { type: 'string' },
            failureAnalysis: { type: 'string' },
            calibrationNotes: { type: 'string' },
            tableDimensions: {
              type: 'object',
              properties: {
                x: { type: 'number' },
                y: { type: 'number' },
                z: { type: 'number' },
              },
              required: ['x', 'y', 'z'],
            },
            recommendedRobotBase: {
              type: 'object',
              properties: { x: { type: 'number' }, z: { type: 'number' } },
              required: ['x', 'z'],
            },
            targetObject: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                shape: { type: 'string' },
                position: {
                  type: 'object',
                  properties: {
                    x: { type: 'number' },
                    y: { type: 'number' },
                    z: { type: 'number' },
                  },
                  required: ['x', 'y', 'z'],
                },
                dimensions: {
                  type: 'object',
                  properties: {
                    x: { type: 'number' },
                    y: { type: 'number' },
                    z: { type: 'number' },
                  },
                  required: ['x', 'y', 'z'],
                },
                colorHex: { type: 'string' },
                massKg: { type: 'number' },
                friction: { type: 'number' },
                materialLabel: { type: 'string' },
              },
              required: [
                'name',
                'shape',
                'position',
                'dimensions',
                'colorHex',
                'massKg',
                'friction',
                'materialLabel',
              ],
            },
            obstacles: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  shape: { type: 'string' },
                  position: {
                    type: 'object',
                    properties: {
                      x: { type: 'number' },
                      y: { type: 'number' },
                      z: { type: 'number' },
                    },
                    required: ['x', 'y', 'z'],
                  },
                  dimensions: {
                    type: 'object',
                    properties: {
                      x: { type: 'number' },
                      y: { type: 'number' },
                      z: { type: 'number' },
                    },
                    required: ['x', 'y', 'z'],
                  },
                  colorHex: { type: 'string' },
                  massKg: { type: 'number' },
                  friction: { type: 'number' },
                  materialLabel: { type: 'string' },
                },
                required: [
                  'name',
                  'shape',
                  'position',
                  'dimensions',
                  'colorHex',
                  'massKg',
                  'friction',
                  'materialLabel',
                ],
              },
            },
            spatialZones: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  label: { type: 'string' },
                  kind: { type: 'string' },
                  centerX: { type: 'number' },
                  centerZ: { type: 'number' },
                  radiusM: { type: 'number' },
                  notes: { type: 'string' },
                },
                required: ['label', 'kind', 'centerX', 'centerZ', 'radiusM', 'notes'],
              },
            },
            mjcfXml: { type: 'string' },
            initialControllerPython: { type: 'string' },
            controllerParams: {
              type: 'object',
              properties: {
                approachHeightM: { type: 'number' },
                preGraspOffsetX: { type: 'number' },
                preGraspOffsetY: { type: 'number' },
                descentSpeedMs: { type: 'number' },
                kpStiffness: { type: 'number' },
                kdDamping: { type: 'number' },
                gripForceN: { type: 'number' },
                gripRampSteps: { type: 'number' },
                clearanceArcM: { type: 'number' },
                liftHeightM: { type: 'number' },
              },
              required: [
                'approachHeightM',
                'preGraspOffsetX',
                'preGraspOffsetY',
                'descentSpeedMs',
                'kpStiffness',
                'kdDamping',
                'gripForceN',
                'gripRampSteps',
                'clearanceArcM',
                'liftHeightM',
              ],
            },
            randomizations: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'number' },
                  friction: { type: 'number' },
                  mass: { type: 'number' },
                  xyOffset: { type: 'array', items: { type: 'number' } },
                },
                required: ['id', 'friction', 'mass', 'xyOffset'],
              },
            },
          },
          required: [
            'hasRobotInVideo',
            'executionOutcome',
            'outcomeSummary',
            'detectedRobotDescription',
            'failureAnalysis',
            'calibrationNotes',
            'tableDimensions',
            'recommendedRobotBase',
            'targetObject',
            'obstacles',
            'spatialZones',
            'mjcfXml',
            'initialControllerPython',
            'controllerParams',
            'randomizations',
          ],
        },
      };

      const carbonContentParts: Array<
        Awaited<ReturnType<typeof prepareMediaForInteractions>> | { type: 'text'; text: string }
      > = [primaryMediaPart];
      if (secondaryMediaPart) {
        carbonContentParts.push(secondaryMediaPart);
      }
      carbonContentParts.push({
        type: 'text',
        text:
          `Analyze this real-world footage for target platform '${robotSpec.name}' and task goal: "${record.state.taskGoal}".\n` +
          (secondaryMediaPart
            ? `MULTI-VIEW INPUT: Two camera views are provided in order (View 1: Primary Workcell Camera, View 2: Secondary / Wrist / Alternate Camera). Fuse both camera angles to triangulate metric 3D poses, resolve occlusions, and evaluate grasp contact accurately.\n`
            : '') +
          `1) FIRST determine honestly whether an actual robotic arm, gripper, or robot is visibly present ('hasRobotInVideo': true/false) and describe it in 'detectedRobotDescription' (or state "No robotic arm detected — environment-only capture").\n` +
          `2) CLASSIFY 'executionOutcome' strictly as one of three values: "no_robot" (if hasRobotInVideo is false), "success" (if a robot is present and visibly completes the manipulation task cleanly without dropping, colliding, or failing), or "failure" (if a robot is present and visibly fails — collision, tip-over, slip, missed grasp, stuck, or incomplete task). Provide a concise 1–2 sentence verdict in 'outcomeSummary', and put detailed physical root-cause diagnosis (if failure) or grasp robustness & clearance margin analysis (if success) in 'failureAnalysis'.\n` +
          `3) Calibrate metric 3D coordinates (meters, origin x=0,y=0,z=0 near front-center tabletop edge; tabletop surface at y=0; primary target object placed in x in [0.16, 0.34], z in [-0.18, 0.18]).\n` +
          `4) Reconstruct the primary target object and ALL notable nearby tabletop/scene obstacles (1 to 7 objects, each with shape one of: mug, cylinder, box, bottle, sphere), plus 2–4 'spatialZones' (kind: 'robot_mount', 'clear_zone', or 'hazard_zone') and 'recommendedRobotBase' ({x, z}) so the 2D top-down spatial map is accurate and ready for simulation.\n` +
          `5) Generate valid simulation-ready MuJoCo MJCF XML (<mujoco>) containing 'target_body', 'target_object' geom, and all mapped obstacles (omit robot joints in the scene XML if hasRobotInVideo is false).\n` +
          `6) Emit a baseline Python controller ('def execute_policy(model, data, step):'), numeric 'controllerParams', and 20 'randomizations' (ids 1..20). If executionOutcome is "success", tune the baseline controllerParams to reflect the successful nominal demonstration so the 20-env swarm can stress-test its robustness margins under friction/mass/pose perturbations.`,
      });

      const carbonPrompt = [
        {
          type: 'user_input',
          content: carbonContentParts,
        },
      ];

      const carbonTurn = await createLoggedInteraction({
        stage: 'carbon_reconstruct',
        model: APP_CONFIG.models.spatialArchitect.id,
        thinkingLevel: APP_CONFIG.models.spatialArchitect.thinkingLevel,
        input: carbonPrompt,
        systemInstruction:
          'You are a robotics perception, 3D scene reconstruction, and MuJoCo physics architect. Be strictly truthful about both robot presence (hasRobotInVideo) and whether the robot succeeded or failed (executionOutcome: "success" | "failure" | "no_robot"). Return strictly valid JSON matching the requested schema.',
        responseFormat: carbonResponseFormat,
        background: true,
        waitForCompletion: true,
        timeoutMs: APP_CONFIG.timeouts.carbonStageMs,
        signal: abortController.signal,
        onPollHeartbeat: (status, elapsedMs) => {
          const sec = Math.max(1, Math.round(elapsedMs / 1000));
          updateStage(record, 'carbon_reconstruct', {
            status: 'active',
            detail: hasSecondaryMedia
              ? `Gemini 4 Carbon fusing 2 camera views (${status} · ${sec}s elapsed)...`
              : `Gemini 4 Carbon calibrating metric 3D scene (${status} · ${sec}s elapsed)...`,
          });
        },
      });

      const parsedCarbon = parseStructuredJson<SceneSpec>(carbonTurn.text);
      const validShapes = new Set(['cylinder', 'box', 'mug', 'bottle', 'sphere']);
      const sanitizeShape = (s: string) =>
        (validShapes.has(s) ? s : 'cylinder') as SceneSpec['targetObject']['shape'];

      const hasRobotInVideo = Boolean(parsedCarbon.hasRobotInVideo);
      const rawOutcome = String(parsedCarbon.executionOutcome ?? '').toLowerCase().trim();
      const executionOutcome: ExecutionOutcome = !hasRobotInVideo
        ? 'no_robot'
        : rawOutcome === 'success'
          ? 'success'
          : 'failure';

      const sceneMode: SceneMode =
        !hasRobotInVideo && !forceSimulateRobot
          ? 'scene_map_only'
          : executionOutcome === 'success'
            ? 'robot_success_verification'
            : 'robot_failure_repair';

      const outcomeSummary =
        (parsedCarbon.outcomeSummary && parsedCarbon.outcomeSummary.trim()) ||
        (executionOutcome === 'no_robot'
          ? 'Environment-only capture with no robot arm detected; simulation-ready spatial map generated.'
          : executionOutcome === 'success'
            ? 'Nominal robot execution succeeded; launching 20-environment robustness stress test.'
            : 'Manipulation failure observed in source footage; launching closed-loop policy repair.');

      const recommendedBase = {
        x: Math.max(-0.18, Math.min(0.18, parsedCarbon.recommendedRobotBase?.x ?? 0)),
        z: Math.max(-0.18, Math.min(0.18, parsedCarbon.recommendedRobotBase?.z ?? 0)),
      };

      sceneSpec = {
        hasRobotInVideo,
        executionOutcome,
        outcomeSummary,
        sceneMode,
        detectedRobotDescription:
          parsedCarbon.detectedRobotDescription ||
          (hasRobotInVideo ? robotSpec.name : 'No robotic arm detected — environment-only capture'),
        failureAnalysis: parsedCarbon.failureAnalysis,
        calibrationNotes: parsedCarbon.calibrationNotes,
        tableDimensions: {
          x: Math.max(0.5, Math.min(1.4, parsedCarbon.tableDimensions?.x ?? 0.82)),
          y: Math.max(0.02, Math.min(0.1, parsedCarbon.tableDimensions?.y ?? 0.04)),
          z: Math.max(0.42, Math.min(1.1, parsedCarbon.tableDimensions?.z ?? 0.62)),
        },
        recommendedRobotBase: recommendedBase,
        targetObject: {
          ...parsedCarbon.targetObject,
          shape: sanitizeShape(parsedCarbon.targetObject.shape),
          position: {
            x: Math.max(0.12, Math.min(0.36, parsedCarbon.targetObject.position.x)),
            y: Math.max(0.03, parsedCarbon.targetObject.position.y),
            z: Math.max(-0.22, Math.min(0.22, parsedCarbon.targetObject.position.z)),
          },
          dimensions: {
            x: Math.max(0.04, Math.min(0.16, parsedCarbon.targetObject.dimensions.x)),
            y: Math.max(0.05, Math.min(0.2, parsedCarbon.targetObject.dimensions.y)),
            z: Math.max(0.04, Math.min(0.16, parsedCarbon.targetObject.dimensions.z)),
          },
        },
        obstacles: (parsedCarbon.obstacles ?? []).slice(0, 7).map((obs) => ({
          ...obs,
          shape: sanitizeShape(obs.shape),
          position: {
            x: Math.max(-0.18, Math.min(0.44, obs.position.x)),
            y: Math.max(0.03, obs.position.y),
            z: Math.max(-0.28, Math.min(0.28, obs.position.z)),
          },
          dimensions: {
            x: Math.max(0.035, Math.min(0.18, obs.dimensions.x)),
            y: Math.max(0.04, Math.min(0.26, obs.dimensions.y)),
            z: Math.max(0.035, Math.min(0.18, obs.dimensions.z)),
          },
        })),
        spatialZones: normalizeSpatialZones(parsedCarbon.spatialZones, recommendedBase),
        mjcfXml: parsedCarbon.mjcfXml,
        initialControllerPython: parsedCarbon.initialControllerPython,
        controllerParams: parsedCarbon.controllerParams,
        randomizations: normalizeRandomizations(
          parsedCarbon.randomizations,
          parsedCarbon.targetObject.friction ?? 0.45,
          parsedCarbon.targetObject.massKg ?? 0.24,
        ),
      };

      lastInteractionId = carbonTurn.id;
      currentPolicyCode = sceneSpec.initialControllerPython;
      currentControllerParams = { ...sceneSpec.controllerParams };

      record.state.sceneSpec = sceneSpec;
      setPipelineProgress(record, {
        percent: sceneMode === 'scene_map_only' ? 100 : progCfg.carbonTargetPct,
        targetPercent: sceneMode === 'scene_map_only' ? 100 : progCfg.swarmStartPct,
        expectedStageDurationSec: 2,
        stageLabel:
          sceneMode === 'scene_map_only'
            ? 'Scene Map & MuJoCo XML Ready'
            : 'Stage 1/4 · Scene Calibrated',
      });
      updateStage(record, 'carbon_reconstruct', {
        status: 'completed',
        detail:
          sceneMode === 'scene_map_only'
            ? `No robot detected — compiled simulation-ready 3D scene & 2D spatial map (${1 + sceneSpec.obstacles.length} mapped objects).`
            : sceneMode === 'robot_success_verification'
              ? `Successful demonstration detected — calibrated ${sceneSpec.targetObject.name} + ${sceneSpec.obstacles.length} obstacle(s) for 20-env robustness stress test.`
              : `Failure detected — reconstructed ${sceneSpec.targetObject.name} + ${sceneSpec.obstacles.length} obstacle(s) and 20 MuJoCo repair variants.`,
        interactionId: carbonTurn.id,
        previousInteractionId: null,
      });
      appendTrace(record, {
        id: carbonTurn.id,
        previousInteractionId: null,
        stage: 'carbon_reconstruct',
        model: APP_CONFIG.models.spatialArchitect.id,
        thinkingLevel: APP_CONFIG.models.spatialArchitect.thinkingLevel,
        title:
          sceneMode === 'scene_map_only'
            ? 'Scene Map & Simulation-Ready MJCF Compilation'
            : sceneMode === 'robot_success_verification'
              ? 'Successful Demo Reconstruction & Stress-Test Setup'
              : 'Spatial Real-to-Sim Failure Reconstruction',
        thoughtSummary: carbonTurn.thoughtSummary || sceneSpec.calibrationNotes,
        outputExcerpt: `${sceneSpec.outcomeSummary} — ${sceneSpec.failureAnalysis}`,
        durationMs: carbonTurn.durationMs,
        timestamp: new Date().toISOString(),
      });
      broadcastSessionEvent(record, { type: 'scene_ready', sceneSpec });

      // If no robot was found in the video, finish cleanly in Scene Map Mode!
      if (sceneMode === 'scene_map_only') {
        updateStage(record, 'swarm_hydrogen_triage', {
          status: 'skipped',
          detail: 'Skipped (no robot arm in video). Scene map & MuJoCo XML are ready for simulation.',
        });
        updateStage(record, 'neon_policy_patch', {
          status: 'skipped',
          detail: 'Skipped in Scene Map Mode. Click "Mount Robot & Simulate Policy" anytime to test a robot here.',
        });
        updateStage(record, 'neon_long_decode', {
          status: 'skipped',
          detail: 'Skipped in Scene Map Mode. Export the compiled MuJoCo XML or JSON map directly.',
        });
        record.state.status = 'completed';
        broadcastSessionEvent(record, { type: 'pipeline_complete', session: record.state });
        return;
      }
    }

    // -------------------------------------------------------------------------
    // STAGE 2 & 3: 20-ENV SWARM + HYDROGEN (low) <-> NEON (xhigh) REPAIR LOOP
    // -------------------------------------------------------------------------
    let previousPolicyCode: string | null = null;
    for (let iteration = 1; iteration <= APP_CONFIG.simulation.maxRepairIterations; iteration++) {
      const useExternal = Boolean(record.state.webhook.enabled && record.state.webhook.url);
      const isFollowUpIter = iteration > 1;
      const followUpBase = Math.min(
        85,
        progCfg.neonTargetPct + (iteration - 2) * progCfg.reverifySpanPerIterPct,
      );
      const swarmFloor = isFollowUpIter ? followUpBase : progCfg.swarmStartPct;
      const swarmCeiling = isFollowUpIter ? Math.min(86, followUpBase + 2) : progCfg.swarmTargetPct;
      const hydrogenFloor = isFollowUpIter ? Math.min(86, followUpBase + 2) : progCfg.hydrogenStartPct;
      const hydrogenCeiling = isFollowUpIter
        ? Math.min(87, followUpBase + 4)
        : progCfg.hydrogenTargetPct;

      setPipelineProgress(record, {
        percent: swarmFloor,
        targetPercent: swarmCeiling,
        expectedStageDurationSec: useExternal ? 14 : progCfg.swarmExpectedSec,
        stageLabel: `Stage 2/4 · 20-Env Swarm (Iter ${iteration})`,
      });
      updateStage(record, 'swarm_hydrogen_triage', {
        status: 'active',
        detail: useExternal
          ? `Iteration ${iteration}: Fanning out 20 environments to Cloud MuJoCo webhook...`
          : `Iteration ${iteration}: Evaluating 20 domain-randomized environments...`,
        previousInteractionId: lastInteractionId,
      });

      const swarmResult = useExternal
        ? await executeExternalSwarmWebhook({
            webhookUrl: record.state.webhook.url,
            authToken: record.webhookAuthToken,
            sceneSpec,
            controllerCode: currentPolicyCode,
            controllerParams: currentControllerParams,
            robotModel: record.state.robotModel,
          })
        : evaluateBuiltInSwarm({
            sceneSpec,
            controllerParams: currentControllerParams,
            robotModel: record.state.robotModel,
            rolloutDurationSec: record.state.rolloutDurationSec,
          });

      const compactTelemetry = swarmResult.rollouts.map((r) => ({
        envId: r.envId,
        success: r.success,
        finalObjectZ: r.finalObjectZ,
        peakContactForceN: r.peakContactForceN,
        maxSlipVelocityMs: r.maxSlipVelocityMs,
        failureReason: r.failureReason,
        params: r.params,
      }));

      setPipelineProgress(record, {
        percent: hydrogenFloor,
        targetPercent: hydrogenCeiling,
        expectedStageDurationSec: progCfg.hydrogenExpectedSec,
        stageLabel: `Stage 2/4 · Gemini 4 Hydrogen Triage (${swarmResult.passed}/${swarmResult.total})`,
      });
      updateStage(record, 'swarm_hydrogen_triage', {
        status: 'active',
        detail: `Iteration ${iteration}: ${swarmResult.passed}/${swarmResult.total} passed — clustering anomalies with Gemini 4 Hydrogen...`,
        previousInteractionId: lastInteractionId,
      });

      const isSuccessVerification = sceneSpec.sceneMode === 'robot_success_verification';

      const hydrogenTurn = await createLoggedInteraction({
        stage: `swarm_hydrogen_triage_iter_${iteration}`,
        model: APP_CONFIG.models.telemetryAggregator.id,
        thinkingLevel: APP_CONFIG.models.telemetryAggregator.thinkingLevel,
        previousInteractionId: lastInteractionId,
        timeoutMs: APP_CONFIG.timeouts.hydrogenStageMs,
        signal: abortController.signal,
        input:
          `Mode: ${isSuccessVerification ? 'Robustness Stress Test of a Successful Demonstration' : 'Closed-Loop Failure Repair'}\n` +
          `Iteration ${iteration} 20-Environment MuJoCo Rollout Telemetry (${swarmResult.passed}/${swarmResult.total} passed):\n` +
          `${JSON.stringify(compactTelemetry, null, 2)}\n\n` +
          `Current Controller Parameters:\n${JSON.stringify(currentControllerParams, null, 2)}\n\n` +
          `Cluster any failure modes (force spikes > 45N, slip velocity > 0.08 m/s, insufficient Z-lift < 0.12m), identify the worst-case environment IDs, and summarize safety margins and actionable parameter bounds concisely.`,
      });

      const hydrogenPrevId = lastInteractionId;
      lastInteractionId = hydrogenTurn.id;

      updateStage(record, 'swarm_hydrogen_triage', {
        status: 'completed',
        detail: `Iteration ${iteration}: Triaged ${swarmResult.passed}/${swarmResult.total} rollouts (${Math.round((swarmResult.passed / swarmResult.total) * 100)}%).`,
        interactionId: hydrogenTurn.id,
        previousInteractionId: hydrogenPrevId,
      });
      appendTrace(record, {
        id: hydrogenTurn.id,
        previousInteractionId: hydrogenPrevId,
        stage: 'swarm_hydrogen_triage',
        model: APP_CONFIG.models.telemetryAggregator.id,
        thinkingLevel: APP_CONFIG.models.telemetryAggregator.thinkingLevel,
        iteration,
        title: `Iteration ${iteration} ${isSuccessVerification ? 'Robustness' : 'Telemetry'} Triage (${swarmResult.passed}/${swarmResult.total})`,
        thoughtSummary: hydrogenTurn.thoughtSummary,
        outputExcerpt: hydrogenTurn.text.slice(0, 500),
        durationMs: hydrogenTurn.durationMs,
        timestamp: new Date().toISOString(),
      });

      if (
        swarmResult.passed === swarmResult.total ||
        iteration === APP_CONFIG.simulation.maxRepairIterations
      ) {
        const finalIterRecord: IterationRecord = {
          iteration,
          passed: swarmResult.passed,
          total: swarmResult.total,
          executionSource: useExternal ? 'external_webhook' : 'builtin_physics',
          rollouts: swarmResult.rollouts,
          telemetrySummary: hydrogenTurn.text,
          rootCauseFix:
            swarmResult.passed === swarmResult.total
              ? isSuccessVerification && iteration === 1
                ? 'Demonstrated policy passed all 20/20 domain-randomized robustness environments on the first pass (force <= 45 N, slip <= 0.08 m/s, lift >= 0.12 m) — no repair patch required.'
                : 'All 20/20 domain-randomized environments satisfied contact-force (<=45 N), slip (<=0.08 m/s), and lift-height (>=0.12 m) verification gates.'
              : null,
          policyCode: currentPolicyCode,
          previousPolicyCode,
          controllerParams: { ...currentControllerParams },
          hydrogenInteractionId: hydrogenTurn.id,
          neonInteractionId: null,
        };
        record.state.iterations = [...record.state.iterations, finalIterRecord];
        broadcastSessionEvent(record, { type: 'iteration_complete', iteration: finalIterRecord });

        if (swarmResult.passed === swarmResult.total) {
          updateStage(record, 'neon_policy_patch', {
            status: 'completed',
            detail:
              isSuccessVerification && iteration === 1
                ? 'Verified at Iteration 1: 20/20 robustness stress-test variants passed (no patch needed).'
                : `Converged at Iteration ${iteration}: 20/20 environments verified (100%).`,
          });
          break;
        }
        break;
      }

      // Escalate to Gemini 4 Neon (`xhigh` thinking) to patch policy
      const neonFloor = isFollowUpIter ? Math.min(86, followUpBase + 4) : progCfg.neonStartPct;
      const neonCeiling = isFollowUpIter
        ? Math.min(87, followUpBase + progCfg.reverifySpanPerIterPct)
        : progCfg.neonTargetPct;
      setPipelineProgress(record, {
        percent: neonFloor,
        targetPercent: neonCeiling,
        expectedStageDurationSec: progCfg.neonExpectedSec,
        stageLabel: `Stage 3/4 · Gemini 4 Neon Repair (Iter ${iteration})`,
      });
      updateStage(record, 'neon_policy_patch', {
        status: 'active',
        detail: `Iteration ${iteration}: Escalating ${swarmResult.total - swarmResult.passed} failed rollout(s) to Gemini 4 Neon (xhigh)...`,
        previousInteractionId: lastInteractionId,
      });

      const neonResponseFormat = {
        type: 'text',
        mime_type: 'application/json',
        schema: {
          type: 'object',
          properties: {
            rootCauseFix: { type: 'string' },
            patchedControllerPython: { type: 'string' },
            patchedControllerParams: {
              type: 'object',
              properties: {
                approachHeightM: { type: 'number' },
                preGraspOffsetX: { type: 'number' },
                preGraspOffsetY: { type: 'number' },
                descentSpeedMs: { type: 'number' },
                kpStiffness: { type: 'number' },
                kdDamping: { type: 'number' },
                gripForceN: { type: 'number' },
                gripRampSteps: { type: 'number' },
                clearanceArcM: { type: 'number' },
                liftHeightM: { type: 'number' },
              },
              required: [
                'approachHeightM',
                'preGraspOffsetX',
                'preGraspOffsetY',
                'descentSpeedMs',
                'kpStiffness',
                'kdDamping',
                'gripForceN',
                'gripRampSteps',
                'clearanceArcM',
                'liftHeightM',
              ],
            },
          },
          required: ['rootCauseFix', 'patchedControllerPython', 'patchedControllerParams'],
        },
      };

      const neonTurn = await createLoggedInteraction({
        stage: `neon_policy_patch_iter_${iteration}`,
        model: APP_CONFIG.models.policyPatcher.id,
        thinkingLevel: APP_CONFIG.models.policyPatcher.thinkingLevel,
        previousInteractionId: lastInteractionId,
        background: true,
        waitForCompletion: true,
        timeoutMs: APP_CONFIG.timeouts.neonStageMs,
        signal: abortController.signal,
        onPollHeartbeat: (status, elapsedMs) => {
          const sec = Math.max(1, Math.round(elapsedMs / 1000));
          updateStage(record, 'neon_policy_patch', {
            status: 'active',
            detail: `Iteration ${iteration}: Gemini 4 Neon patching policy (${status} · ${sec}s elapsed)...`,
          });
        },
        systemInstruction:
          'You are an adversarial sim-to-real robotics control engineer. Rewrite execute_policy(model, data, step) and tune patchedControllerParams so ALL 20 randomized environments pass (peak contact force <= 45 N, slip velocity <= 0.08 m/s, final Z lift >= 0.12 m). Return strictly valid JSON.',
        input:
          `Hydrogen Telemetry Diagnosis (Iteration ${iteration}, ${swarmResult.passed}/${swarmResult.total} passed):\n${hydrogenTurn.text}\n\n` +
          `Current Controller Parameters:\n${JSON.stringify(currentControllerParams, null, 2)}\n\n` +
          `Physics Verification Constraints:\n` +
          `- Clearance: approachHeightM + clearanceArcM should exceed max obstacle height + 0.05m (e.g., approachHeightM ~0.18-0.24, clearanceArcM ~0.08-0.12).\n` +
          `- Contact compliance: kpStiffness <= 210, kdDamping >= 24, descentSpeedMs <= 0.14, gripRampSteps >= 75.\n` +
          `- Anti-slip: gripForceN in [18, 28] N with preGraspOffsetX=0, preGraspOffsetY=0 centered on estimated target.\n` +
          `- Lift: liftHeightM >= 0.15 m.\n\n` +
          `Return JSON with rootCauseFix, patchedControllerPython, and patchedControllerParams.`,
        responseFormat: neonResponseFormat,
      });

      const neonPrevId = lastInteractionId;
      lastInteractionId = neonTurn.id;
      const patchPayload = parseStructuredJson<{
        rootCauseFix: string;
        patchedControllerPython: string;
        patchedControllerParams: ControllerParams;
      }>(neonTurn.text);

      const iterRecord: IterationRecord = {
        iteration,
        passed: swarmResult.passed,
        total: swarmResult.total,
        executionSource: useExternal ? 'external_webhook' : 'builtin_physics',
        rollouts: swarmResult.rollouts,
        telemetrySummary: hydrogenTurn.text,
        rootCauseFix: patchPayload.rootCauseFix,
        policyCode: currentPolicyCode,
        previousPolicyCode,
        controllerParams: { ...currentControllerParams },
        hydrogenInteractionId: hydrogenTurn.id,
        neonInteractionId: neonTurn.id,
      };
      record.state.iterations = [...record.state.iterations, iterRecord];
      broadcastSessionEvent(record, { type: 'iteration_complete', iteration: iterRecord });

      previousPolicyCode = currentPolicyCode;
      currentPolicyCode = patchPayload.patchedControllerPython;
      currentControllerParams = patchPayload.patchedControllerParams;

      updateStage(record, 'neon_policy_patch', {
        status: 'active',
        detail: `Iteration ${iteration} patched by Neon — re-evaluating 20-environment swarm...`,
        interactionId: neonTurn.id,
        previousInteractionId: neonPrevId,
      });
      appendTrace(record, {
        id: neonTurn.id,
        previousInteractionId: neonPrevId,
        stage: 'neon_policy_patch',
        model: APP_CONFIG.models.policyPatcher.id,
        thinkingLevel: APP_CONFIG.models.policyPatcher.thinkingLevel,
        iteration,
        title: `Iteration ${iteration} Policy Patch`,
        thoughtSummary: neonTurn.thoughtSummary,
        outputExcerpt: patchPayload.rootCauseFix,
        durationMs: neonTurn.durationMs,
        timestamp: new Date().toISOString(),
      });
    }

    // -------------------------------------------------------------------------
    // STAGE 4: LONG DECODE — Asynchronous Deployment Bundle (background=True)
    // -------------------------------------------------------------------------
    setPipelineProgress(record, {
      percent: progCfg.longDecodeStartPct,
      targetPercent: progCfg.longDecodeTargetPct,
      expectedStageDurationSec: progCfg.longDecodeExpectedSec,
      stageLabel: 'Stage 4/4 · Compiling Deployment Bundle',
    });
    updateStage(record, 'neon_long_decode', {
      status: 'active',
      detail: 'Launching asynchronous Long Decode (background=true) for deployment bundle...',
      previousInteractionId: lastInteractionId,
    });

    const bundlePrevId = lastInteractionId;
    const bundleTurn = await createLoggedInteraction({
      stage: 'neon_long_decode_bundle',
      model: APP_CONFIG.models.bundleCompiler.id,
      thinkingLevel: APP_CONFIG.models.bundleCompiler.thinkingLevel,
      previousInteractionId: bundlePrevId,
      background: true,
      timeoutMs: APP_CONFIG.timeouts.longDecodeDispatchMs,
      signal: abortController.signal,
      input:
        `Compile the complete, production-ready Sim-to-Real deployment package for '${robotSpec.name}' and task "${record.state.taskGoal}":\n` +
        `1) A fenced \`\`\`python\`\`\` block with the standalone LeRobot / MuJoCo controller script including torque limits (${robotSpec.maxTorqueNm} Nm), impedance gains, and emergency stop guards.\n` +
        `2) A fenced \`\`\`xml\`\`\` block containing the verified MJCF regression suite with domain randomization annotations.\n` +
        `3) A comprehensive Markdown Sim-to-Real Verification & Contact Dynamics Audit Report detailing each iteration's pass-rate trajectory, contact force peaks, and real-hardware deployment checklist.`,
    });

    const initialSections = splitBundleSections(
      bundleTurn.text,
      currentPolicyCode,
      sceneSpec.mjcfXml,
    );

    const bundleStatus =
      bundleTurn.status === 'completed'
        ? 'completed'
        : bundleTurn.status === 'failed'
          ? 'failed'
          : 'in_progress';

    const bundle: DeploymentBundle = {
      interactionId: bundleTurn.id,
      status: bundleStatus,
      standaloneControllerPy: initialSections.standaloneControllerPy,
      regressionSuiteXml: initialSections.regressionSuiteXml,
      auditReportMarkdown:
        initialSections.auditReportMarkdown ||
        '# Deployment Bundle Compiling\n\nLong Decode interaction is running asynchronously in the background. Use Refresh Status to fetch the finalized output.',
      rawText: bundleTurn.text,
      updatedAt: new Date().toISOString(),
    };

    record.state.bundle = bundle;
    record.state.status = 'completed';
    setPipelineProgress(record, {
      percent: 100,
      targetPercent: 100,
      expectedStageDurationSec: 1,
      stageLabel: 'Verified · 4/4 Stages Complete',
    });
    updateStage(record, 'neon_long_decode', {
      status: bundleStatus === 'completed' ? 'completed' : 'active',
      detail:
        bundleStatus === 'completed'
          ? 'Long Decode deployment bundle compiled and ready for export.'
          : 'Long Decode running asynchronously in background; bundle polls automatically.',
      interactionId: bundleTurn.id,
      previousInteractionId: bundlePrevId,
    });
    appendTrace(record, {
      id: bundleTurn.id,
      previousInteractionId: bundlePrevId,
      stage: 'neon_long_decode',
      model: APP_CONFIG.models.bundleCompiler.id,
      thinkingLevel: APP_CONFIG.models.bundleCompiler.thinkingLevel,
      title: 'Long Decode Deployment Bundle',
      thoughtSummary: bundleTurn.thoughtSummary || 'Asynchronous Long Decode job dispatched.',
      outputExcerpt: `Interaction ${bundleTurn.id} (${bundleStatus})`,
      durationMs: bundleTurn.durationMs,
      timestamp: new Date().toISOString(),
    });

    broadcastSessionEvent(record, { type: 'bundle_update', bundle });
    broadcastSessionEvent(record, { type: 'pipeline_complete', session: record.state });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unexpected error in Gemini 4 pipeline.';
    // Only overwrite status if cancelFailureToPolicyPipeline hasn't already finalized cancellation
    if (record.state.status === 'running') {
      record.state.status = 'error';
      record.state.error = message;

      record.state.stages = record.state.stages.map((s) =>
        s.status === 'active' ? { ...s, status: 'error', detail: message } : s,
      );
      broadcastSessionEvent(record, {
        type: 'stage_update',
        stages: record.state.stages,
        status: 'error',
        progress: record.state.progress,
      });
      broadcastSessionEvent(record, { type: 'pipeline_error', message });
    }
  } finally {
    if (record.abortController === abortController) {
      record.abortController = null;
    }
    record.isPipelineRunning = false;
  }
}

/**
 * Cancels an in-flight Failure-to-Policy pipeline run immediately by aborting
 * its AbortController, resetting the session state, and broadcasting a halted
 * SSE event so the UI unlocks without waiting for any pending Gemini request.
 */
export function cancelFailureToPolicyPipeline(record: InternalSessionRecord): boolean {
  logFunctionCall('cancelFailureToPolicyPipeline', {
    sessionId: record.state.id,
    wasRunning: record.isPipelineRunning,
  });

  if (!record.isPipelineRunning && record.state.status !== 'running') {
    return false;
  }

  const cancelMessage = 'Pipeline cancelled by user. Click Analyze & Verify Policy to run again.';
  record.abortController?.abort();
  record.abortController = null;
  record.isPipelineRunning = false;

  record.state.status = 'error';
  record.state.error = cancelMessage;
  record.state.stages = record.state.stages.map((s) =>
    s.status === 'active' ? { ...s, status: 'error', detail: cancelMessage } : s,
  );
  if (record.state.progress) {
    record.state.progress = {
      ...record.state.progress,
      targetPercent: record.state.progress.percent,
      stageLabel: 'Cancelled by user',
      stageUpdatedAt: new Date().toISOString(),
    };
  }

  broadcastSessionEvent(record, {
    type: 'stage_update',
    stages: record.state.stages,
    status: 'error',
    progress: record.state.progress,
  });
  broadcastSessionEvent(record, { type: 'pipeline_error', message: cancelMessage });
  return true;
}

/**
 * Polls the background Long Decode interaction from Gemini 4 Neon and updates
 * the session's DeploymentBundle once finished.
 */
export async function refreshLongDecodeBundle(
  record: InternalSessionRecord,
): Promise<DeploymentBundle | null> {
  logFunctionCall('refreshLongDecodeBundle', {
    sessionId: record.state.id,
    bundleInteractionId: record.state.bundle?.interactionId ?? null,
  });

  if (!record.state.bundle) {
    return null;
  }

  const latest = await getLoggedInteraction(record.state.bundle.interactionId);
  const lastIter = record.state.iterations[record.state.iterations.length - 1];
  const fallbackPolicy =
    lastIter?.policyCode ?? record.state.sceneSpec?.initialControllerPython ?? '';
  const fallbackXml = record.state.sceneSpec?.mjcfXml ?? '';

  const mappedStatus: DeploymentBundle['status'] =
    latest.status === 'completed'
      ? 'completed'
      : latest.status === 'failed'
        ? 'failed'
        : 'in_progress';

  const sections = latest.text
    ? splitBundleSections(latest.text, fallbackPolicy, fallbackXml)
    : {
        standaloneControllerPy: record.state.bundle.standaloneControllerPy,
        regressionSuiteXml: record.state.bundle.regressionSuiteXml,
        auditReportMarkdown: record.state.bundle.auditReportMarkdown,
      };

  const updated: DeploymentBundle = {
    interactionId: record.state.bundle.interactionId,
    status: mappedStatus,
    standaloneControllerPy: sections.standaloneControllerPy,
    regressionSuiteXml: sections.regressionSuiteXml,
    auditReportMarkdown: sections.auditReportMarkdown,
    rawText: latest.text || record.state.bundle.rawText,
    updatedAt: new Date().toISOString(),
  };

  record.state.bundle = updated;
  if (mappedStatus === 'completed') {
    updateStage(record, 'neon_long_decode', {
      status: 'completed',
      detail: 'Long Decode deployment bundle compiled and ready for export.',
    });
  } else if (mappedStatus === 'failed') {
    updateStage(record, 'neon_long_decode', {
      status: 'error',
      detail: 'Background Long Decode failed. Retry compilation.',
    });
  }
  broadcastSessionEvent(record, { type: 'bundle_update', bundle: updated });
  return updated;
}
