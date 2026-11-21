/**
 * Interactive Dynamic 3D Robot Simulation Viewport (`RobotStage3D`)
 *
 * Renders the composite Three.js articulated robot arm (`SO-ARM101` or
 * `Franka Panda`), dynamically scaled metric workcell, Gemini Carbon's
 * reconstructed target object and obstacles, real-time animated trajectory
 * spline, expanding contact-impact ripple ring, and live contact-force vector
 * arrow inside a resizable, full-bleed stage container.
 *
 * Dynamic Interactive Capabilities:
 * - **Live Physics Sandbox (`livePhysicsSandbox.ts`):** Continuously executes a
 *   real-time rigid-body + analytical 6-DOF IK pick-lift-return loop even before
 *   video upload, reacting instantly to tabletop drags, disturbance impulses
 *   ("Nudge"), and live Friction / Mass / Clearance tuning.
 * - **Direct 3D Tabletop Drag-and-Drop (`stageInteractionController.ts`):**
 *   Click and drag the target object or any obstacle directly across the 3D
 *   workbench to test collision corridors, obstacle knocks, and real-time IK
 *   re-planning. Grabbing an object during Replay automatically transitions into
 *   Live Physics mode.
 * - **60-Frame Rollout Replay:** Replays any of the 20 domain-randomized
 *   environments (`E01`–`E20`) emitted by Gemini 4 iterations with synchronized
 *   scrubber and telemetry chart tracking.
 * - **Optional Dual View (`stageViewportRenderer.ts`):** Automatically unlocks
 *   single-context split rendering (`Dual Cam` / `Dual Arms`) when 2 videos are
 *   uploaded and stays cleanly in `Single` view when 1 video is staged.
 *
 * Use cases:
 * - Hands-on stress testing of calibrated MuJoCo scenes by dragging obstacles,
 *   nudging the target object, or dialing down surface friction in real time.
 * - Inspecting 20-environment failure vs. repaired trajectories side-by-side
 *   with uploaded source video(s) via PiP.
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  Activity,
  Camera,
  ChevronDown,
  ChevronUp,
  Circle,
  Columns2,
  Download,
  Film,
  Grid3x3,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RotateCcw,
  Route,
  SlidersHorizontal,
  Square,
  Video,
  Wind,
  Zap,
} from 'lucide-react';
import {
  APP_CONFIG,
  type CameraPresetId,
  type RobotModelId,
  type SecondaryCameraModeId,
  type StageViewModeId,
} from '../config/appConfig';
import type { RolloutResult, SceneSpec } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { formatElapsedSeconds } from '../utils/pipelineProgress';
import { resolveTaskGoalPose } from '../utils/taskGoalResolver';
import {
  createLivePhysicsSandbox,
  type LivePhysicsParams,
  type LivePhysicsSandbox,
  type LiveSimPhase,
} from '../three/livePhysicsSandbox';
import { buildRobotArmRig, type RobotRigHandles } from '../three/robotRigBuilder';
import {
  attachStageInteractionController,
  type DraggableObjectEntry,
} from '../three/stageInteractionController';
import {
  startStageClipRecording,
  type StageClipRecorderHandle,
} from '../three/stageVideoRecorder';
import { renderRobotStageViewports } from '../three/stageViewportRenderer';
import {
  buildSceneObjectMesh,
  buildTrajectoryLine,
  createContactShadowTexture,
  createWorkcellGridTexture,
} from '../three/workcellSceneBuilder';

interface RobotStage3DProps {
  robotModel: RobotModelId;
  sceneSpec: SceneSpec | null;
  activeRollout: RolloutResult | null;
  sourceVideoUrl: string | null;
  localVideoPreviewUrl?: string | null;
  sourceMimeType: string | null;
  hasSecondaryVideo?: boolean;
  secondaryVideoUrl?: string | null;
  localSecondaryVideoPreviewUrl?: string | null;
  secondaryMimeType?: string | null;
  iterationNumber: number | null;
  currentFrameIndex: number;
  hideRobotArm?: boolean;
  onFrameIndexChange: (index: number) => void;
  onClipExported?: (message: string) => void;
}

interface LiveHudReadout {
  timeSec: number;
  phase: LiveSimPhase;
  contactForceN: number;
  slipVelocityMs: number;
  objectHeightM: number;
  faultState: 'ok' | 'collision' | 'slip';
  tasksCompleted: number;
}

const DEFAULT_STANDBY_SPEC: SceneSpec = {
  hasRobotInVideo: true,
  executionOutcome: 'failure',
  outcomeSummary: 'Live interactive physics sandbox — drag objects or upload video to compile.',
  sceneMode: 'robot_failure_repair',
  detectedRobotDescription: 'Standby calibration rig',
  failureAnalysis: 'Standby calibration workcell awaiting smartphone video ingestion.',
  calibrationNotes: 'Default SO-ARM101 tabletop calibration reference.',
  tableDimensions: { x: 0.74, y: 0.04, z: 0.56 },
  recommendedRobotBase: { x: 0, z: 0 },
  targetObject: {
    name: 'Calibration Ceramic Mug',
    shape: 'mug',
    position: { x: 0.23, y: 0.048, z: 0.02 },
    dimensions: { x: 0.078, y: 0.096, z: 0.078 },
    colorHex: '#e2e8f0',
    massKg: 0.28,
    friction: 0.44,
    materialLabel: 'Glazed Ceramic',
  },
  obstacles: [
    {
      name: 'Reference Bottle',
      shape: 'bottle',
      position: { x: 0.24, y: 0.085, z: -0.11 },
      dimensions: { x: 0.068, y: 0.17, z: 0.068 },
      colorHex: '#334155',
      massKg: 0.42,
      friction: 0.35,
      materialLabel: 'Brushed Steel',
    },
  ],
  spatialZones: [],
  mjcfXml: '',
  initialControllerPython: '',
  controllerParams: {
    approachHeightM: 0.18,
    preGraspOffsetX: 0,
    preGraspOffsetY: 0,
    descentSpeedMs: 0.12,
    kpStiffness: 180,
    kdDamping: 26,
    gripForceN: 24,
    gripRampSteps: 80,
    clearanceArcM: 0.11,
    liftHeightM: 0.16,
  },
  randomizations: [],
};

const LIVE_SPLINE_SAMPLES = 36;

/**
 * Probes WebGL2 renderer string to detect software rasterizers (SwiftShader,
 * llvmpipe, WARP) for adaptive frame pacing without altering scene contents.
 */
function detectSoftwareWebGL(): boolean {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) return true;
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
  const gpu = String(gl.getParameter(param));
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
}

/**
 * Interactive Three.js 3D viewport component supporting both real-time dynamic
 * rigid-body IK simulation and synchronized 60-frame rollout replay.
 */
export const RobotStage3D: React.FC<RobotStage3DProps> = ({
  robotModel,
  sceneSpec,
  activeRollout,
  sourceVideoUrl,
  localVideoPreviewUrl = null,
  sourceMimeType,
  hasSecondaryVideo = false,
  secondaryVideoUrl = null,
  localSecondaryVideoPreviewUrl = null,
  secondaryMimeType = null,
  iterationNumber,
  currentFrameIndex,
  hideRobotArm = false,
  onFrameIndexChange,
  onClipExported,
}) => {
  const canvasMountRef = useRef<HTMLDivElement | null>(null);
  const canvasElementRef = useRef<HTMLCanvasElement | null>(null);
  const recorderHandleRef = useRef<StageClipRecorderHandle | null>(null);
  const [simMode, setSimMode] = useState<'live' | 'replay'>('live');
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 1 | 2>(1);
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>('perspective');
  const [stageLayout, setStageLayout] = useState<StageViewModeId>(
    hasSecondaryVideo ? 'dual_cam' : 'single',
  );
  const [secondaryCamMode, setSecondaryCamMode] = useState<SecondaryCameraModeId>('wrist');
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showForceVector, setShowForceVector] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [hudCollapsed, setHudCollapsed] = useState(false);
  const [showPhysicsPanel, setShowPhysicsPanel] = useState(false);
  const [showRecordPanel, setShowRecordPanel] = useState(false);
  const [recordDurationSec, setRecordDurationSec] = useState<number>(
    APP_CONFIG.simulation.videoExport.defaultDurationSec,
  );
  const [isRecordingClip, setIsRecordingClip] = useState(false);
  const [recordElapsedSec, setRecordElapsedSec] = useState(0);
  const [recordError, setRecordError] = useState<string | null>(null);
  const [expandedStage, setExpandedStage] = useState(false);
  const [showSourcePip, setShowSourcePip] = useState(false);
  const [pipVideoError, setPipVideoError] = useState(false);
  const [pipSecondaryVideoError, setPipSecondaryVideoError] = useState(false);
  const [viewMode, setViewMode] = useState<'3d' | 'cloud_mp4'>('3d');
  const [contextLost, setContextLost] = useState(false);

  // Live sandbox interactive physics parameters
  const [liveParams, setLiveParams] = useState<LivePhysicsParams>({
    friction: APP_CONFIG.simulation.liveSandbox.defaultFriction,
    massKg: APP_CONFIG.simulation.liveSandbox.defaultMassKg,
    clearanceM: APP_CONFIG.simulation.liveSandbox.defaultClearanceM,
    gripForceN: APP_CONFIG.simulation.liveSandbox.defaultGripForceN,
  });
  const [liveHud, setLiveHud] = useState<LiveHudReadout>({
    timeSec: 0,
    phase: 'approach',
    contactForceN: 0,
    slipVelocityMs: 0,
    objectHeightM: 0.048,
    faultState: 'ok',
    tasksCompleted: 0,
  });
  const [hoveredBodyName, setHoveredBodyName] = useState<string | null>(null);
  const [draggedBodyName, setDraggedBodyName] = useState<string | null>(null);

  const effectiveSourceVideoUrl = localVideoPreviewUrl || sourceVideoUrl;
  const effectiveSecondaryVideoUrl = localSecondaryVideoPreviewUrl || secondaryVideoUrl;
  const effectiveStageLayout: StageViewModeId = hasSecondaryVideo ? stageLayout : 'single';
  const effectiveSimMode: 'live' | 'replay' =
    activeRollout && simMode === 'replay' ? 'replay' : 'live';
  const secondaryRobotModel: RobotModelId =
    robotModel === 'so_arm101' ? 'franka_panda' : 'so_arm101';

  // Refs shared with the persistent Three.js animation loop (prevents canvas remount)
  const isPlayingRef = useRef(isPlaying);
  const speedRef = useRef(playbackSpeed);
  const robotModelRef = useRef<RobotModelId>(robotModel);
  const simModeRef = useRef<'live' | 'replay'>(effectiveSimMode);
  const stageLayoutRef = useRef<StageViewModeId>(effectiveStageLayout);
  const secondaryCamModeRef = useRef<SecondaryCameraModeId>(secondaryCamMode);
  const hideRobotArmRef = useRef(hideRobotArm);
  const showTrajectoryRef = useRef(showTrajectory);
  const showForceRef = useRef(showForceVector);
  const showGridRef = useRef(showGrid);
  const activeRolloutRef = useRef<RolloutResult | null>(activeRollout);
  const sceneSpecRef = useRef<SceneSpec>(sceneSpec ?? DEFAULT_STANDBY_SPEC);
  const liveParamsRef = useRef<LivePhysicsParams>(liveParams);
  const frameProgressRef = useRef(currentFrameIndex);
  const onFrameChangeRef = useRef(onFrameIndexChange);
  const viewportSizeRef = useRef({ width: 960, height: 480 });
  const draggedBodyIdRef = useRef<string | null>(null);
  const hoveredBodyIdRef = useRef<string | null>(null);
  const cameraGoalRef = useRef<{
    active: boolean;
    pos: THREE.Vector3;
    target: THREE.Vector3;
  }>({
    active: false,
    pos: new THREE.Vector3(...APP_CONFIG.cameraPresets.perspective.position),
    target: new THREE.Vector3(...APP_CONFIG.cameraPresets.perspective.target),
  });

  // Scene object refs updated imperatively when props change
  const sceneRef = useRef<THREE.Scene | null>(null);
  const robotRigRef = useRef<RobotRigHandles | null>(null);
  const secondaryRigRef = useRef<RobotRigHandles | null>(null);
  const dynamicGroupRef = useRef<THREE.Group | null>(null);
  const targetMeshRef = useRef<THREE.Group | null>(null);
  const obstacleMeshMapRef = useRef<Map<string, THREE.Group>>(new Map());
  const draggablesRef = useRef<DraggableObjectEntry[]>([]);
  const sandboxRef = useRef<LivePhysicsSandbox>(
    createLivePhysicsSandbox(sceneSpec ?? DEFAULT_STANDBY_SPEC),
  );
  const replayTrajectoryLineRef = useRef<THREE.Line | null>(null);
  const liveTrajectoryLineRef = useRef<THREE.Line | null>(null);
  const liveSplinePointsRef = useRef<THREE.Vector3[]>(
    Array.from({ length: LIVE_SPLINE_SAMPLES }, () => new THREE.Vector3()),
  );
  const forceArrowRef = useRef<THREE.ArrowHelper | null>(null);
  const impactRingRef = useRef<THREE.Mesh | null>(null);
  const goalPadGroupRef = useRef<THREE.Group | null>(null);
  const goalPadRingMatRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const hoverRingRef = useRef<THREE.Mesh | null>(null);
  const gridPlaneRef = useRef<THREE.Mesh | null>(null);
  const tableBaseRef = useRef<THREE.Mesh | null>(null);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);
  useEffect(() => {
    speedRef.current = playbackSpeed;
  }, [playbackSpeed]);
  useEffect(() => {
    robotModelRef.current = robotModel;
  }, [robotModel]);
  useEffect(() => {
    simModeRef.current = effectiveSimMode;
  }, [effectiveSimMode]);
  useEffect(() => {
    setStageLayout(hasSecondaryVideo ? 'dual_cam' : 'single');
  }, [hasSecondaryVideo]);
  useEffect(() => {
    stageLayoutRef.current = effectiveStageLayout;
  }, [effectiveStageLayout]);
  useEffect(() => {
    secondaryCamModeRef.current = secondaryCamMode;
  }, [secondaryCamMode]);
  useEffect(() => {
    hideRobotArmRef.current = hideRobotArm;
  }, [hideRobotArm]);
  useEffect(() => {
    showTrajectoryRef.current = showTrajectory;
  }, [showTrajectory]);
  useEffect(() => {
    showForceRef.current = showForceVector;
    if (forceArrowRef.current && !showForceVector) forceArrowRef.current.visible = false;
  }, [showForceVector]);
  useEffect(() => {
    showGridRef.current = showGrid;
    if (gridPlaneRef.current) gridPlaneRef.current.visible = showGrid;
  }, [showGrid]);
  useEffect(() => {
    liveParamsRef.current = liveParams;
  }, [liveParams]);

  // Auto-switch into Replay whenever a new rollout arrives, and back to Live when cleared
  useEffect(() => {
    activeRolloutRef.current = activeRollout;
    frameProgressRef.current = 0;
    if (activeRollout) {
      setSimMode('replay');
    } else {
      setSimMode('live');
    }
    if (!activeRollout?.previewVideoUrl && viewMode === 'cloud_mp4') {
      setViewMode('3d');
    }
  }, [activeRollout]);

  // Seed live physics sliders from newly calibrated Gemini Carbon sceneSpec
  useEffect(() => {
    const spec = sceneSpec ?? DEFAULT_STANDBY_SPEC;
    sceneSpecRef.current = spec;
    setLiveParams({
      friction: Number(
        Math.max(0.15, Math.min(1.0, spec.targetObject.friction ?? 0.44)).toFixed(2),
      ),
      massKg: Number(
        Math.max(0.1, Math.min(1.2, spec.targetObject.massKg ?? 0.28)).toFixed(2),
      ),
      clearanceM: Number(
        Math.max(0.03, Math.min(0.2, spec.controllerParams.clearanceArcM ?? 0.11)).toFixed(2),
      ),
      gripForceN: Number(
        Math.max(10, Math.min(38, spec.controllerParams.gripForceN ?? 24)).toFixed(0),
      ),
    });
  }, [sceneSpec]);

  useEffect(() => {
    onFrameChangeRef.current = onFrameIndexChange;
  }, [onFrameIndexChange]);
  useEffect(() => {
    setPipVideoError(false);
  }, [effectiveSourceVideoUrl]);
  useEffect(() => {
    setPipSecondaryVideoError(false);
  }, [effectiveSecondaryVideoUrl]);

  // Sync manual scrubber clicks into the 3D playback cursor
  useEffect(() => {
    if (Math.abs(frameProgressRef.current - currentFrameIndex) > 1.5) {
      frameProgressRef.current = currentFrameIndex;
    }
  }, [currentFrameIndex]);

  /**
   * Triggers a smoothly damped camera transition to a named preset.
   */
  const handleSelectCameraPreset = (presetId: CameraPresetId): void => {
    logFunctionCall('handleSelectCameraPreset', { presetId });
    setCameraPreset(presetId);
    const preset = APP_CONFIG.cameraPresets[presetId];
    cameraGoalRef.current = {
      active: true,
      pos: new THREE.Vector3(...preset.position),
      target: new THREE.Vector3(...preset.target),
    };
  };

  /**
   * Injects a real-time impulse into the target object and switches into Live Physics mode.
   */
  const handleNudgeObject = (): void => {
    logFunctionCall('RobotStage3D.handleNudgeObject');
    if (simModeRef.current !== 'live') {
      setSimMode('live');
      simModeRef.current = 'live';
    }
    setIsPlaying(true);
    sandboxRef.current.applyNudgeImpulse();
  };

  /**
   * Restores all tabletop objects and physics defaults back to calibrated home poses.
   */
  const handleResetSandbox = (): void => {
    logFunctionCall('RobotStage3D.handleResetSandbox');
    const spec = sceneSpecRef.current;
    sandboxRef.current.resetBodies();
    setLiveParams({
      friction: Number(
        Math.max(0.15, Math.min(1.0, spec.targetObject.friction ?? 0.44)).toFixed(2),
      ),
      massKg: Number(
        Math.max(0.1, Math.min(1.2, spec.targetObject.massKg ?? 0.28)).toFixed(2),
      ),
      clearanceM: Number(
        Math.max(0.03, Math.min(0.2, spec.controllerParams.clearanceArcM ?? 0.11)).toFixed(2),
      ),
      gripForceN: Number(
        Math.max(10, Math.min(38, spec.controllerParams.gripForceN ?? 24)).toFixed(0),
      ),
    });
  };

  // Ensure any active WebGL recording cleanly aborts on component unmount
  useEffect(() => {
    return () => {
      if (recorderHandleRef.current) {
        recorderHandleRef.current.cancel();
        recorderHandleRef.current = null;
      }
    };
  }, []);

  /**
   * Starts recording a real-time WebM clip (e.g., 15s, 30s, or 60s / 1 minute)
   * directly from the active 3D WebGL stage canvas and downloads it on completion.
   */
  const handleStartClipRecording = (targetSec: number): void => {
    const canvas = canvasElementRef.current;
    if (!canvas) {
      setRecordError('3D simulation canvas is not ready yet.');
      return;
    }
    logFunctionCall('RobotStage3D.handleStartClipRecording', {
      targetSec,
      simMode: effectiveSimMode,
      robotModel,
    });
    setRecordError(null);
    setViewMode('3d');
    setIsPlaying(true);
    if (effectiveSimMode === 'replay') {
      frameProgressRef.current = 0;
      onFrameIndexChange(0);
    }

    const modeTag = effectiveSimMode === 'replay' ? 'replay' : 'live';
    const handle = startStageClipRecording({
      canvas,
      targetDurationSec: targetSec,
      filenameBase: `lettherebesims_${robotModel}_${modeTag}_${Math.round(targetSec)}s`,
      onTick: (elapsed) => setRecordElapsedSec(elapsed),
      onComplete: (filename, sizeBytes, actualSec) => {
        recorderHandleRef.current = null;
        setIsRecordingClip(false);
        setRecordElapsedSec(0);
        const mb = (sizeBytes / (1024 * 1024)).toFixed(1);
        onClipExported?.(`Saved ${filename} (${Math.round(actualSec)}s · ${mb} MB).`);
      },
      onError: (msg) => {
        recorderHandleRef.current = null;
        setIsRecordingClip(false);
        setRecordElapsedSec(0);
        setRecordError(msg);
      },
    });

    if (handle) {
      recorderHandleRef.current = handle;
      setIsRecordingClip(true);
      setRecordElapsedSec(0);
    }
  };

  /**
   * Stops an active WebGL clip recording early and downloads the captured video.
   */
  const handleStopClipRecording = (): void => {
    logFunctionCall('RobotStage3D.handleStopClipRecording', { recordElapsedSec });
    if (recorderHandleRef.current) {
      recorderHandleRef.current.stop();
    }
  };

  // 1. Mount persistent Three.js WebGL renderer once
  useEffect(() => {
    const container = canvasMountRef.current;
    if (!container) return;

    logFunctionCall('RobotStage3D.initWebGL', {
      width: container.clientWidth,
      height: container.clientHeight,
    });

    const isSoftwareGl = detectSoftwareWebGL();
    const width = Math.max(320, container.clientWidth);
    const height = Math.max(240, container.clientHeight);
    viewportSizeRef.current = { width, height };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090e);
    scene.fog = new THREE.FogExp2(0x07090e, 0.36);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.02, 12);
    camera.position.set(...APP_CONFIG.cameraPresets.perspective.position);

    const secondaryCamera = new THREE.PerspectiveCamera(
      APP_CONFIG.secondaryCameraModes.wrist.fov,
      (width * 0.5) / height,
      0.015,
      12,
    );

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    canvasElementRef.current = renderer.domElement;

    // WebGL context loss/restore guards
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      setContextLost(true);
    };
    const handleContextRestored = () => setContextLost(false);
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored);

    // Studio IBL reflections via RoomEnvironment
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;
    pmrem.dispose();

    // Warm/cool 3-point laboratory lighting rig
    const hemiLight = new THREE.HemisphereLight(0xbfd7ff, 0x161d2c, 0.85);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff4e0, 2.2);
    keyLight.position.set(1.1, 1.8, 1.2);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 0.2;
    keyLight.shadow.camera.far = 5;
    const d = 0.65;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.9);
    rimLight.position.set(-1.0, 0.9, -1.1);
    scene.add(rimLight);

    // Calibrated metric tabletop & floor base (dynamically scales to sceneSpec.tableDimensions)
    const tableBaseGeo = new THREE.BoxGeometry(0.86, 0.036, 0.66);
    const tableBaseMat = new THREE.MeshStandardMaterial({
      color: 0x0f1523,
      roughness: 0.72,
      metalness: 0.2,
    });
    const tableBase = new THREE.Mesh(tableBaseGeo, tableBaseMat);
    tableBase.position.set(0.14, -0.019, 0);
    tableBase.receiveShadow = true;
    scene.add(tableBase);
    tableBaseRef.current = tableBase;

    const gridTex = createWorkcellGridTexture(renderer.capabilities.getMaxAnisotropy());
    const gridPlaneGeo = new THREE.PlaneGeometry(0.82, 0.62);
    const gridPlaneMat = new THREE.MeshStandardMaterial({
      map: gridTex,
      roughness: 0.65,
      metalness: 0.15,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const gridPlane = new THREE.Mesh(gridPlaneGeo, gridPlaneMat);
    gridPlane.rotation.x = -Math.PI / 2;
    gridPlane.position.set(0.14, 0.0005, 0);
    gridPlane.receiveShadow = true;
    gridPlane.visible = showGridRef.current;
    scene.add(gridPlane);
    gridPlaneRef.current = gridPlane;

    // Subtle grounding contact shadow beneath robot pedestal
    const shadowTex = createContactShadowTexture();
    const baseShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(0.24, 0.24),
      new THREE.MeshBasicMaterial({
        map: shadowTex,
        transparent: true,
        depthWrite: false,
        opacity: 0.55,
      }),
    );
    baseShadow.rotation.x = -Math.PI / 2;
    baseShadow.position.set(0, 0.001, 0);
    scene.add(baseShadow);

    // Coordinate axes origin indicator (small precision triad)
    const axes = new THREE.AxesHelper(0.09);
    axes.position.set(-0.22, 0.003, -0.24);
    scene.add(axes);

    // Dynamic group for target object, obstacles, and replay trajectory spline
    const dynamicGroup = new THREE.Group();
    scene.add(dynamicGroup);
    dynamicGroupRef.current = dynamicGroup;

    // Live Dynamic Trajectory Preview Line (recomputed in-place as objects move)
    const liveLinePositions = new Float32Array(LIVE_SPLINE_SAMPLES * 3);
    const liveLineGeo = new THREE.BufferGeometry();
    liveLineGeo.setAttribute('position', new THREE.BufferAttribute(liveLinePositions, 3));
    const liveLineMat = new THREE.LineBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.82,
    });
    const liveTrajectoryLine = new THREE.Line(liveLineGeo, liveLineMat);
    liveTrajectoryLine.frustumCulled = false;
    scene.add(liveTrajectoryLine);
    liveTrajectoryLineRef.current = liveTrajectoryLine;

    // Expanding Contact Impact Shockwave Ring
    const impactRingGeo = new THREE.RingGeometry(0.022, 0.032, 36);
    const impactRingMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const impactRing = new THREE.Mesh(impactRingGeo, impactRingMat);
    impactRing.rotation.x = -Math.PI / 2;
    impactRing.position.set(0.22, 0.003, 0);
    impactRing.visible = false;
    scene.add(impactRing);
    impactRingRef.current = impactRing;

    // Composite 3D Goal Placement Pad (Destination Target Zone for Full Task Completion)
    const goalPadGroup = new THREE.Group();
    const goalBaseGeo = new THREE.CylinderGeometry(0.056, 0.06, 0.0025, 36);
    const goalBaseMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.35,
      metalness: 0.72,
    });
    const goalBaseMesh = new THREE.Mesh(goalBaseGeo, goalBaseMat);
    goalBaseMesh.position.y = 0.0014;
    goalPadGroup.add(goalBaseMesh);

    const goalOuterRingGeo = new THREE.RingGeometry(0.046, 0.055, 36);
    const goalOuterRingMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    const goalOuterRing = new THREE.Mesh(goalOuterRingGeo, goalOuterRingMat);
    goalOuterRing.rotation.x = -Math.PI / 2;
    goalOuterRing.position.y = 0.0032;
    goalPadGroup.add(goalOuterRing);

    const goalInnerRingGeo = new THREE.RingGeometry(0.016, 0.022, 28);
    const goalInnerRingMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
    });
    const goalInnerRing = new THREE.Mesh(goalInnerRingGeo, goalInnerRingMat);
    goalInnerRing.rotation.x = -Math.PI / 2;
    goalInnerRing.position.y = 0.0033;
    goalPadGroup.add(goalInnerRing);

    goalPadGroup.position.set(0.22, 0, -0.16);
    scene.add(goalPadGroup);
    goalPadGroupRef.current = goalPadGroup;
    goalPadRingMatRef.current = goalOuterRingMat;

    // Interactive Hover / Drag Selection Halo Ring
    const hoverRingGeo = new THREE.RingGeometry(0.044, 0.052, 36);
    const hoverRingMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });
    const hoverRing = new THREE.Mesh(hoverRingGeo, hoverRingMat);
    hoverRing.rotation.x = -Math.PI / 2;
    hoverRing.visible = false;
    scene.add(hoverRing);
    hoverRingRef.current = hoverRing;

    // Contact Force Vector Arrow
    const forceArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0.2, 0.1, 0),
      0.08,
      0xf59e0b,
      0.022,
      0.014,
    );
    forceArrow.visible = false;
    scene.add(forceArrow);
    forceArrowRef.current = forceArrow;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...APP_CONFIG.cameraPresets.perspective.target);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.25;
    controls.maxDistance = 1.9;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.update();

    controls.addEventListener('start', () => {
      cameraGoalRef.current.active = false;
    });

    // Attach Raycast Hover & Tabletop Drag-and-Drop Controller
    const detachInteraction = attachStageInteractionController({
      domElement: renderer.domElement,
      camera,
      controls,
      handlers: {
        getDraggables: () => draggablesRef.current,
        getLayoutMode: () => stageLayoutRef.current,
        isInteractiveEnabled: () => true,
        onDragStart: (bodyId) => {
          draggedBodyIdRef.current = bodyId;
          // Seamlessly transition into Live Physics mode whenever the user grabs an object
          if (simModeRef.current !== 'live') {
            simModeRef.current = 'live';
            setSimMode('live');
          }
          const sandbox = sandboxRef.current;
          const body =
            bodyId === sandbox.targetBody.id
              ? sandbox.targetBody
              : sandbox.obstacleBodies.find((b) => b.id === bodyId);
          setDraggedBodyName(body?.name ?? bodyId);
        },
        onDragMove: (bodyId, worldX, worldZ) => {
          sandboxRef.current.setBodyPositionFromDrag(bodyId, worldX, worldZ);
        },
        onDragEnd: (bodyId) => {
          draggedBodyIdRef.current = null;
          setDraggedBodyName(null);
          sandboxRef.current.finishBodyDrag(bodyId);
        },
        onHoverChange: (bodyId) => {
          hoveredBodyIdRef.current = bodyId;
          if (!bodyId) {
            setHoveredBodyName(null);
            return;
          }
          const sandbox = sandboxRef.current;
          const body =
            bodyId === sandbox.targetBody.id
              ? sandbox.targetBody
              : sandbox.obstacleBodies.find((b) => b.id === bodyId);
          setHoveredBodyName(body?.name ?? null);
        },
      },
    });

    // ResizeObserver for clean container-matched viewport sizing (including Expand/Compact toggle)
    const resizeObserver = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect || rect.width < 10 || rect.height < 10) return;
      viewportSizeRef.current = { width: rect.width, height: rect.height };
      renderer.setSize(rect.width, rect.height);
    });
    resizeObserver.observe(container);

    // Pre-allocated math helpers (zero GC inside animation loop)
    const _forceDir = new THREE.Vector3(0, 1, 0);
    const _forceOrigin = new THREE.Vector3();
    const clock = new THREE.Clock();
    let frameId = 0;
    let lastReportedFrame = -1;
    let lastHudPushMs = 0;
    let disposed = false;

    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const tick = () => {
      if (disposed) return;
      const delta = Math.min(0.05, clock.getDelta());
      const nowMs = performance.now();

      // Smooth camera preset damping
      if (cameraGoalRef.current.active) {
        const goal = cameraGoalRef.current;
        camera.position.x = THREE.MathUtils.damp(camera.position.x, goal.pos.x, 6, delta);
        camera.position.y = THREE.MathUtils.damp(camera.position.y, goal.pos.y, 6, delta);
        camera.position.z = THREE.MathUtils.damp(camera.position.z, goal.pos.z, 6, delta);
        controls.target.x = THREE.MathUtils.damp(controls.target.x, goal.target.x, 6, delta);
        controls.target.y = THREE.MathUtils.damp(controls.target.y, goal.target.y, 6, delta);
        controls.target.z = THREE.MathUtils.damp(controls.target.z, goal.target.z, 6, delta);
        if (
          camera.position.distanceTo(goal.pos) < 0.004 &&
          controls.target.distanceTo(goal.target) < 0.004
        ) {
          goal.active = false;
        }
      }
      controls.update();

      // Smoothly scale tabletop workcell to match calibrated SceneSpec dimensions
      const spec = sceneSpecRef.current;
      if (tableBaseRef.current && gridPlaneRef.current) {
        const targetScaleX = Math.max(0.75, Math.min(1.5, spec.tableDimensions.x / 0.86));
        const targetScaleZ = Math.max(0.75, Math.min(1.5, spec.tableDimensions.z / 0.66));
        tableBaseRef.current.scale.x = THREE.MathUtils.damp(
          tableBaseRef.current.scale.x,
          targetScaleX,
          5,
          delta,
        );
        tableBaseRef.current.scale.z = THREE.MathUtils.damp(
          tableBaseRef.current.scale.z,
          targetScaleZ,
          5,
          delta,
        );
        gridPlaneRef.current.scale.x = tableBaseRef.current.scale.x;
        gridPlaneRef.current.scale.y = tableBaseRef.current.scale.z;
      }

      const rig = robotRigRef.current;
      const secondaryRig = secondaryRigRef.current;
      const rollout = activeRolloutRef.current;
      const targetMesh = targetMeshRef.current;
      const arrow = forceArrowRef.current;
      const liveLine = liveTrajectoryLineRef.current;
      const replayLine = replayTrajectoryLineRef.current;
      const ring = impactRingRef.current;
      const hover = hoverRingRef.current;

      const useReplay = simModeRef.current === 'replay' && rollout && rollout.trajectory.length > 0;

      if (useReplay && rig && rollout) {
        // MODE A: Real-Time Synchronized Rollout Replay (matches full clip / rollout duration)
        if (liveLine) liveLine.visible = false;
        if (replayLine) replayLine.visible = showTrajectoryRef.current;

        const frames = rollout.trajectory;
        const totalDurationSec = Math.max(
          0.5,
          frames[frames.length - 1]?.timeSec ?? rollout.durationSec ?? 3.0,
        );
        const framesPerSecond = Math.max(0.5, (frames.length - 1) / totalDurationSec);
        if (isPlayingRef.current) {
          frameProgressRef.current =
            (frameProgressRef.current + delta * framesPerSecond * speedRef.current) %
            frames.length;
        }
        const idx = Math.floor(frameProgressRef.current) % frames.length;
        const nextIdx = Math.min(frames.length - 1, idx + 1);
        const alpha = frameProgressRef.current - Math.floor(frameProgressRef.current);
        const f0 = frames[idx];
        const f1 = frames[nextIdx];

        const interpAngles: [number, number, number, number, number, number] = [
          THREE.MathUtils.lerp(f0.jointAngles[0], f1.jointAngles[0], alpha),
          THREE.MathUtils.lerp(f0.jointAngles[1], f1.jointAngles[1], alpha),
          THREE.MathUtils.lerp(f0.jointAngles[2], f1.jointAngles[2], alpha),
          THREE.MathUtils.lerp(f0.jointAngles[3], f1.jointAngles[3], alpha),
          THREE.MathUtils.lerp(f0.jointAngles[4], f1.jointAngles[4], alpha),
          THREE.MathUtils.lerp(f0.jointAngles[5], f1.jointAngles[5], alpha),
        ];
        const gripOpen = THREE.MathUtils.lerp(f0.gripperOpen, f1.gripperOpen, alpha);
        rig.applyJointState(interpAngles, gripOpen);
        if (secondaryRig) {
          secondaryRig.applyJointState(interpAngles, gripOpen);
        }

        if (targetMesh) {
          targetMesh.position.set(
            THREE.MathUtils.lerp(f0.objectPos.x, f1.objectPos.x, alpha),
            THREE.MathUtils.lerp(f0.objectPos.y, f1.objectPos.y, alpha),
            THREE.MathUtils.lerp(f0.objectPos.z, f1.objectPos.z, alpha),
          );
          targetMesh.rotation.set(0, 0, THREE.MathUtils.lerp(f0.objectTiltRad, f1.objectTiltRad, alpha));
        }

        const force = THREE.MathUtils.lerp(f0.contactForceN, f1.contactForceN, alpha);
        if (arrow) {
          if (showForceRef.current && force > 1.5) {
            arrow.visible = true;
            _forceOrigin.set(
              THREE.MathUtils.lerp(f0.endEffectorPos.x, f1.endEffectorPos.x, alpha),
              THREE.MathUtils.lerp(f0.endEffectorPos.y, f1.endEffectorPos.y, alpha),
              THREE.MathUtils.lerp(f0.endEffectorPos.z, f1.endEffectorPos.z, alpha),
            );
            arrow.position.copy(_forceOrigin);
            _forceDir.set(0.18, 1, 0.12).normalize();
            arrow.setDirection(_forceDir);
            const arrowLen = Math.min(0.24, Math.max(0.04, force / 260));
            arrow.setLength(arrowLen, 0.022, 0.013);
            arrow.setColor(
              force > APP_CONFIG.simulation.maxSafeContactForceN ? 0xf43f5e : 0xf59e0b,
            );
          } else {
            arrow.visible = false;
          }
        }

        // Sync 3D Goal Placement Pad in Replay mode and highlight emerald when the object lands on the pad
        if (goalPadGroupRef.current) {
          const replayGoal = resolveTaskGoalPose(spec);
          goalPadGroupRef.current.visible = !hideRobotArmRef.current;
          goalPadGroupRef.current.position.set(replayGoal.x, 0, replayGoal.z);
          if (goalPadRingMatRef.current) {
            const distToGoal = targetMesh
              ? Math.hypot(
                  targetMesh.position.x - replayGoal.x,
                  targetMesh.position.z - replayGoal.z,
                )
              : 1;
            const isLandedOnGoal =
              distToGoal <= replayGoal.radiusM + 0.03 &&
              (targetMesh ? targetMesh.position.y <= 0.12 : false);
            goalPadRingMatRef.current.color.setHex(isLandedOnGoal ? 0x34d399 : 0x10b981);
            goalPadRingMatRef.current.opacity = isLandedOnGoal
              ? 0.95 + Math.sin(nowMs * 0.012) * 0.05
              : 0.72;
          }
        }

        // Contact shockwave ring on force spikes during replay
        if (ring) {
          if (force > 22) {
            ring.visible = true;
            const pulse = Math.min(1, (force - 22) / 36);
            ring.position.set(
              targetMesh ? targetMesh.position.x : 0.22,
              0.003,
              targetMesh ? targetMesh.position.z : 0,
            );
            const s = 1 + pulse * 1.9;
            ring.scale.set(s, s, 1);
            const mat = ring.material as THREE.MeshBasicMaterial;
            mat.color.setHex(
              force > APP_CONFIG.simulation.maxSafeContactForceN ? 0xf43f5e : 0xf59e0b,
            );
            mat.opacity = Math.min(0.85, pulse * 0.85);
          } else {
            ring.visible = false;
          }
        }

        if (idx !== lastReportedFrame) {
          lastReportedFrame = idx;
          onFrameChangeRef.current(idx);
        }
      } else {
        // MODE B: Real-Time Dynamic Physics & 6-DOF IK Sandbox
        if (replayLine) replayLine.visible = false;

        const sandbox = sandboxRef.current;
        if (isPlayingRef.current || draggedBodyIdRef.current !== null) {
          const simDt = delta * speedRef.current;
          sandbox.step(
            simDt,
            robotModelRef.current,
            liveParamsRef.current,
            spec.controllerParams,
            draggedBodyIdRef.current,
          );
        }

        if (rig && !hideRobotArmRef.current) {
          rig.applyJointState(sandbox.jointAngles, sandbox.gripperOpen);
        }
        if (secondaryRig && !hideRobotArmRef.current) {
          secondaryRig.applyJointState(sandbox.jointAngles, sandbox.gripperOpen);
        }

        // Sync target & obstacle meshes directly from live physics state
        if (targetMesh) {
          targetMesh.position.set(sandbox.targetBody.x, sandbox.targetBody.y, sandbox.targetBody.z);
          targetMesh.rotation.set(sandbox.targetBody.tiltX, 0, sandbox.targetBody.tiltZ);
        }
        for (let i = 0; i < sandbox.obstacleBodies.length; i++) {
          const obs = sandbox.obstacleBodies[i];
          const mesh = obstacleMeshMapRef.current.get(obs.id);
          if (mesh) {
            mesh.position.set(obs.x, obs.y, obs.z);
            mesh.rotation.set(obs.tiltX, 0, obs.tiltZ);
          }
        }

        // Dynamically recompute live 3D approach spline whenever visible
        if (liveLine) {
          const shouldShowLiveLine = showTrajectoryRef.current && !hideRobotArmRef.current;
          liveLine.visible = shouldShowLiveLine;
          if (shouldShowLiveLine) {
            const pts = liveSplinePointsRef.current;
            const n = sandbox.computePreviewSplinePoints(
              liveParamsRef.current.clearanceM,
              spec.controllerParams.liftHeightM || 0.16,
              pts,
            );
            const attr = liveLine.geometry.getAttribute('position') as THREE.BufferAttribute;
            for (let i = 0; i < n; i++) {
              attr.setXYZ(i, pts[i].x, pts[i].y, pts[i].z);
            }
            attr.needsUpdate = true;
            (liveLine.material as THREE.LineBasicMaterial).color.setHex(
              sandbox.telemetry.faultState === 'ok' ? 0x06b6d4 : 0xf43f5e,
            );
          }
        }

        // Live force vector arrow
        const liveForce = sandbox.telemetry.contactForceN;
        if (arrow) {
          if (showForceRef.current && !hideRobotArmRef.current && liveForce > 1.5) {
            arrow.visible = true;
            _forceOrigin.set(
              sandbox.telemetry.endEffectorPos.x,
              sandbox.telemetry.endEffectorPos.y,
              sandbox.telemetry.endEffectorPos.z,
            );
            arrow.position.copy(_forceOrigin);
            _forceDir.set(0.18, 1, 0.12).normalize();
            arrow.setDirection(_forceDir);
            const arrowLen = Math.min(0.24, Math.max(0.04, liveForce / 240));
            arrow.setLength(arrowLen, 0.022, 0.013);
            arrow.setColor(
              liveForce > APP_CONFIG.simulation.maxSafeContactForceN ? 0xf43f5e : 0xf59e0b,
            );
          } else {
            arrow.visible = false;
          }
        }

        // Sync 3D Goal Placement Pad with the sandbox's active destination zone
        if (goalPadGroupRef.current) {
          goalPadGroupRef.current.visible = !hideRobotArmRef.current;
          goalPadGroupRef.current.position.set(
            sandbox.telemetry.goalX,
            0,
            sandbox.telemetry.goalZ,
          );
          if (goalPadRingMatRef.current) {
            const isPlacingOrDone =
              sandbox.telemetry.phase === 'place' || sandbox.telemetry.phase === 'complete';
            goalPadRingMatRef.current.color.setHex(isPlacingOrDone ? 0x34d399 : 0x10b981);
            goalPadRingMatRef.current.opacity = isPlacingOrDone
              ? 0.95 + Math.sin(nowMs * 0.012) * 0.05
              : 0.72;
          }
        }

        // Expanding contact / collision / task-completion ripple ring
        if (ring) {
          const intensity = sandbox.telemetry.impactIntensity;
          if (intensity > 0.03) {
            ring.visible = true;
            ring.position.set(
              sandbox.telemetry.impactPosition.x,
              0.0035,
              sandbox.telemetry.impactPosition.z,
            );
            const scale = 1.0 + (1 - intensity) * 2.8;
            ring.scale.set(scale, scale, 1);
            const mat = ring.material as THREE.MeshBasicMaterial;
            mat.color.setHex(
              sandbox.telemetry.impactKind === 'complete'
                ? 0x10b981
                : sandbox.telemetry.faultState === 'collision'
                  ? 0xf43f5e
                  : 0xf59e0b,
            );
            mat.opacity = intensity * 0.9;
          } else {
            ring.visible = false;
          }
        }

        // Throttle React HUD telemetry updates to ~12 Hz
        if (nowMs - lastHudPushMs > 85) {
          lastHudPushMs = nowMs;
          setLiveHud({
            timeSec: sandbox.telemetry.timeSec,
            phase: sandbox.telemetry.phase,
            contactForceN: sandbox.telemetry.contactForceN,
            slipVelocityMs: sandbox.telemetry.slipVelocityMs,
            objectHeightM: sandbox.telemetry.objectHeightM,
            faultState: sandbox.telemetry.faultState,
            tasksCompleted: sandbox.telemetry.tasksCompleted,
          });
        }
      }

      // Highlight hovered or actively dragged tabletop body with amber selection halo
      if (hover) {
        const focusId = draggedBodyIdRef.current ?? hoveredBodyIdRef.current;
        if (focusId) {
          const entry = draggablesRef.current.find((d) => d.bodyId === focusId);
          if (entry) {
            hover.visible = true;
            hover.position.set(entry.meshGroup.position.x, 0.0028, entry.meshGroup.position.z);
            const ringScale = draggedBodyIdRef.current ? 1.25 : 1.05;
            hover.scale.set(ringScale, ringScale, 1);
            (hover.material as THREE.MeshBasicMaterial).color.setHex(
              draggedBodyIdRef.current ? 0xf59e0b : 0x06b6d4,
            );
          } else {
            hover.visible = false;
          }
        } else {
          hover.visible = false;
        }
      }

      renderRobotStageViewports({
        renderer,
        scene,
        primaryCamera: camera,
        secondaryCamera,
        primaryRig: rig,
        secondaryRig,
        layoutMode: stageLayoutRef.current,
        secondaryCamMode: secondaryCamModeRef.current,
        hideRobotArm: hideRobotArmRef.current,
        viewportWidth: viewportSizeRef.current.width,
        viewportHeight: viewportSizeRef.current.height,
      });

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      detachInteraction();
      resizeObserver.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored);
      tableBaseGeo.dispose();
      tableBaseMat.dispose();
      gridPlaneGeo.dispose();
      gridPlaneMat.dispose();
      gridTex.dispose();
      shadowTex.dispose();
      liveLineGeo.dispose();
      liveLineMat.dispose();
      impactRingGeo.dispose();
      impactRingMat.dispose();
      goalBaseGeo.dispose();
      goalBaseMat.dispose();
      goalOuterRingGeo.dispose();
      goalOuterRingMat.dispose();
      goalInnerRingGeo.dispose();
      goalInnerRingMat.dispose();
      hoverRingGeo.dispose();
      hoverRingMat.dispose();
      envTex.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // 2. Rebuild Primary & Secondary Robot Rigs whenever robotModel switches
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (robotRigRef.current) {
      scene.remove(robotRigRef.current.root);
      robotRigRef.current.dispose();
      robotRigRef.current = null;
    }
    if (secondaryRigRef.current) {
      scene.remove(secondaryRigRef.current.root);
      secondaryRigRef.current.dispose();
      secondaryRigRef.current = null;
    }

    const rig = buildRobotArmRig(robotModel);
    rig.root.visible = !hideRobotArm;
    scene.add(rig.root);
    robotRigRef.current = rig;

    const compareModel: RobotModelId =
      robotModel === 'so_arm101' ? 'franka_panda' : 'so_arm101';
    const compareRig = buildRobotArmRig(compareModel);
    compareRig.root.visible = false;
    scene.add(compareRig.root);
    secondaryRigRef.current = compareRig;

    return () => {
      scene.remove(rig.root);
      rig.dispose();
      if (robotRigRef.current === rig) {
        robotRigRef.current = null;
      }
      scene.remove(compareRig.root);
      compareRig.dispose();
      if (secondaryRigRef.current === compareRig) {
        secondaryRigRef.current = null;
      }
    };
  }, [robotModel]);

  // 2b. Toggle robot visibility cleanly when entering or leaving Scene Map Mode
  useEffect(() => {
    if (robotRigRef.current) {
      robotRigRef.current.root.visible = !hideRobotArm;
    }
    if (secondaryRigRef.current) {
      secondaryRigRef.current.root.visible = false;
    }
  }, [hideRobotArm]);

  // 3. Rebuild Target Object, Obstacles, Live Sandbox, and Replay Spline when sceneSpec or activeRollout changes
  useEffect(() => {
    const dynamicGroup = dynamicGroupRef.current;
    if (!dynamicGroup) return;

    while (dynamicGroup.children.length > 0) {
      const child = dynamicGroup.children[0];
      dynamicGroup.remove(child);
      child.traverse((obj) => {
        if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
        const mat = (obj as THREE.Mesh).material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
    }
    targetMeshRef.current = null;
    replayTrajectoryLineRef.current = null;
    obstacleMeshMapRef.current.clear();
    draggablesRef.current = [];

    const spec = sceneSpec ?? DEFAULT_STANDBY_SPEC;
    sandboxRef.current = createLivePhysicsSandbox(spec);

    const targetObj = buildSceneObjectMesh(spec.targetObject, true);
    const offsetX = activeRollout?.params.xyOffset[0] ?? 0;
    const offsetZ = activeRollout?.params.xyOffset[1] ?? 0;
    targetObj.position.set(
      spec.targetObject.position.x + offsetX,
      spec.targetObject.dimensions.y * 0.5,
      spec.targetObject.position.z + offsetZ,
    );
    dynamicGroup.add(targetObj);
    targetMeshRef.current = targetObj;
    draggablesRef.current.push({ bodyId: 'target', meshGroup: targetObj });

    spec.obstacles.forEach((obs, idx) => {
      const obsMesh = buildSceneObjectMesh(obs, false);
      obsMesh.position.set(obs.position.x, obs.dimensions.y * 0.5, obs.position.z);
      dynamicGroup.add(obsMesh);
      const obsId = `obstacle_${idx}`;
      obstacleMeshMapRef.current.set(obsId, obsMesh);
      draggablesRef.current.push({ bodyId: obsId, meshGroup: obsMesh });
    });

    // Show recommended robot mounting reticle when in robot-free Scene Map Mode
    if (hideRobotArm) {
      const mountPad = new THREE.Mesh(
        new THREE.RingGeometry(0.055, 0.068, 36),
        new THREE.MeshBasicMaterial({
          color: 0xf59e0b,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        }),
      );
      mountPad.rotation.x = -Math.PI / 2;
      mountPad.position.set(
        spec.recommendedRobotBase?.x ?? 0,
        0.0025,
        spec.recommendedRobotBase?.z ?? 0,
      );
      dynamicGroup.add(mountPad);
    }

    if (activeRollout && activeRollout.trajectory.length > 1) {
      const line = buildTrajectoryLine(activeRollout.trajectory, activeRollout.success);
      line.visible = showTrajectoryRef.current && simModeRef.current === 'replay';
      dynamicGroup.add(line);
      replayTrajectoryLineRef.current = line;
    }
  }, [sceneSpec, activeRollout, hideRobotArm]);

  const activeFrame =
    effectiveSimMode === 'replay' && activeRollout && activeRollout.trajectory[currentFrameIndex]
      ? activeRollout.trajectory[currentFrameIndex]
      : null;

  const totalFrames = activeRollout?.trajectory.length ?? APP_CONFIG.simulation.replayFrameCount;

  const stageHeightClass = expandedStage
    ? 'h-[580px] lg:h-[660px]'
    : 'h-[430px] lg:h-[480px]';

  return (
    <div
      className={`relative w-full ${stageHeightClass} bg-[#07090E] border border-[#1E293B] rounded-lg overflow-hidden select-none transition-[height] duration-200`}
    >
      {/* Isolated WebGL canvas mount — sibling to all React HUD overlays */}
      <div
        ref={canvasMountRef}
        className={`w-full h-full cursor-crosshair ${viewMode === 'cloud_mp4' ? 'hidden' : 'block'}`}
      />

      {/* External Cloud MuJoCo Rendered MP4 Viewer (when webhook returns rendered MP4s) */}
      {viewMode === 'cloud_mp4' && activeRollout?.previewVideoUrl && (
        <div className="w-full h-full flex items-center justify-center bg-[#07090E]">
          <video
            src={activeRollout.previewVideoUrl}
            controls
            autoPlay
            loop
            muted
            className="max-w-full max-h-full object-contain"
          />
        </div>
      )}

      {/* WebGL Context Lost Fallback */}
      {contextLost && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#07090E]/95 text-center p-6">
          <p className="font-display text-lg font-semibold text-slate-100">
            WebGL Context Suspended
          </p>
          <p className="mt-1 text-sm text-slate-400">
            Hardware acceleration context was interrupted. Restore tab focus or switch camera view.
          </p>
        </div>
      )}

      {/* Center Split Hairline Divider for Optional Dual Robotic Arm Views (Active when 2 Videos Uploaded) */}
      {viewMode === '3d' && effectiveStageLayout !== 'single' && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute top-0 bottom-0 left-1/2 -translate-x-px w-px bg-[#1E293B] z-10"
        />
      )}

      {/* Minimal Top-Left Viewport Telemetry Readout (View 1) */}
      <div
        className={`pointer-events-none absolute top-3 left-3 z-10 flex flex-col gap-1 bg-[#0B0E17]/90 border border-[#1E293B] rounded-md px-3 py-2 ${
          effectiveStageLayout !== 'single' ? 'max-w-[46%]' : 'max-w-[82%]'
        }`}
      >
        <div className="flex items-center gap-2 text-xs text-slate-300 flex-wrap">
          {effectiveStageLayout !== 'single' && (
            <span className="font-mono-tabular text-[10px] text-amber-400 uppercase tracking-wider">
              View 1
            </span>
          )}
          <span className="font-display font-semibold text-slate-100">
            {hideRobotArm ? 'Simulation-Ready Scene' : APP_CONFIG.robots[robotModel].name}
          </span>
          <span className="text-slate-600" aria-hidden="true">
            ·
          </span>
          <span className="font-mono-tabular text-slate-400">
            {hideRobotArm
              ? 'Scene Map Mode'
              : effectiveSimMode === 'replay' && activeRollout
                ? `Env E${String(activeRollout.envId).padStart(2, '0')} · Iter ${iterationNumber ?? 1}`
                : liveHud.phase === 'complete'
                  ? `Live Physics · TASK COMPLETE ✓ (${liveHud.tasksCompleted})`
                  : `Live Physics · ${liveHud.phase.toUpperCase()}${liveHud.tasksCompleted > 0 ? ` · ${liveHud.tasksCompleted} Done` : ''}`}
          </span>
          {sceneSpec && !hideRobotArm && (
            <>
              <span className="text-slate-600" aria-hidden="true">
                ·
              </span>
              <span
                className={`font-mono-tabular font-medium ${
                  sceneSpec.executionOutcome === 'success'
                    ? 'text-emerald-400'
                    : 'text-amber-400'
                }`}
              >
                {sceneSpec.executionOutcome === 'success' ? '✓ Demo Verified' : 'Failure Repair'}
              </span>
            </>
          )}
          {effectiveSimMode === 'replay' && activeRollout ? (
            <>
              <span className="text-slate-600" aria-hidden="true">
                ·
              </span>
              <span
                className={`font-mono-tabular font-medium ${
                  activeRollout.success ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {activeRollout.success ? '✓ Task Complete' : '✕ Fault'}
              </span>
            </>
          ) : (
            !hideRobotArm && (
              <>
                <span className="text-slate-600" aria-hidden="true">
                  ·
                </span>
                <span
                  className={`font-mono-tabular font-medium ${
                    liveHud.faultState === 'ok' ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {liveHud.faultState === 'ok'
                    ? '✓ Stable'
                    : liveHud.faultState === 'collision'
                      ? '✕ Collision'
                      : '✕ Slip'}
                </span>
              </>
            )
          )}
        </div>

        {activeFrame ? (
          <div className="flex items-center gap-3 text-[11px] font-mono-tabular text-slate-400 flex-wrap">
            <span>
              t={activeFrame.timeSec.toFixed(1)}s /{' '}
              {(
                activeRollout?.trajectory[activeRollout.trajectory.length - 1]?.timeSec ??
                activeRollout?.durationSec ??
                3.0
              ).toFixed(1)}
              s
            </span>
            <span>
              F=
              <strong
                className={
                  activeFrame.contactForceN > APP_CONFIG.simulation.maxSafeContactForceN
                    ? 'text-rose-400'
                    : 'text-amber-400'
                }
              >
                {activeFrame.contactForceN.toFixed(1)} N
              </strong>
            </span>
            <span>
              Z=
              <strong className="text-cyan-400">{activeFrame.objectPos.y.toFixed(3)} m</strong>
            </span>
            <span>
              μ={activeRollout?.params.friction.toFixed(2)} · m={activeRollout?.params.mass.toFixed(2)} kg
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-[11px] font-mono-tabular text-slate-400 flex-wrap">
            {draggedBodyName ? (
              <span className="text-amber-300">Repositioning {draggedBodyName} on tabletop...</span>
            ) : hoveredBodyName ? (
              <span className="text-cyan-300">Drag {hoveredBodyName} to reposition live</span>
            ) : (
              <>
                <span>
                  F=
                  <strong
                    className={
                      liveHud.contactForceN > APP_CONFIG.simulation.maxSafeContactForceN
                        ? 'text-rose-400'
                        : 'text-amber-400'
                    }
                  >
                    {liveHud.contactForceN.toFixed(1)} N
                  </strong>
                </span>
                <span>
                  Z=
                  <strong className="text-cyan-400">{liveHud.objectHeightM.toFixed(3)} m</strong>
                </span>
                <span>
                  μ={liveParams.friction.toFixed(2)} · m={liveParams.massKg.toFixed(2)} kg
                </span>
                <span className="text-slate-500">· Drag objects on tabletop</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* View 2 Header & Secondary Camera Switcher (Only Active when 2 Videos Uploaded & Split Layout Enabled) */}
      {viewMode === '3d' && effectiveStageLayout !== 'single' && !showSourcePip && (
        <div className="absolute top-3 left-[calc(50%+12px)] right-3 z-10 flex items-center justify-between gap-2 pointer-events-none">
          <div className="pointer-events-auto flex items-center gap-2 bg-[#0B0E17]/90 border border-[#1E293B] rounded-md px-2.5 py-1.5">
            <span className="font-mono-tabular text-[10px] text-cyan-400 uppercase tracking-wider">
              View 2
            </span>
            <span className="font-display text-xs font-semibold text-slate-100">
              {effectiveStageLayout === 'dual_arms' && !hideRobotArm
                ? APP_CONFIG.robots[secondaryRobotModel].name
                : hideRobotArm && secondaryCamMode === 'wrist'
                  ? 'Top-Down Calibration'
                  : APP_CONFIG.secondaryCameraModes[secondaryCamMode].label}
            </span>
            {effectiveStageLayout === 'dual_arms' && !hideRobotArm && (
              <span className="font-mono-tabular text-[11px] text-slate-400">
                · Reach {APP_CONFIG.robots[secondaryRobotModel].reachMeters.toFixed(2)}m
              </span>
            )}
          </div>

          {effectiveStageLayout === 'dual_cam' && (
            <div className="pointer-events-auto flex items-center gap-0.5 bg-[#0B0E17]/95 border border-[#1E293B] rounded-md p-0.5">
              {(Object.keys(APP_CONFIG.secondaryCameraModes) as SecondaryCameraModeId[])
                .filter((modeKey) => !(hideRobotArm && modeKey === 'wrist'))
                .map((modeKey) => (
                  <button
                    key={modeKey}
                    type="button"
                    onClick={() => {
                      logFunctionCall('RobotStage3D.selectSecondaryCam', { modeKey });
                      setSecondaryCamMode(modeKey);
                    }}
                    className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                      (hideRobotArm && secondaryCamMode === 'wrist' ? 'top' : secondaryCamMode) ===
                      modeKey
                        ? 'bg-[#1E293B] text-cyan-300'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {modeKey === 'wrist'
                      ? 'Wrist'
                      : modeKey === 'top'
                        ? 'Top'
                        : modeKey === 'side'
                          ? 'Side'
                          : 'Gripper'}
                  </button>
                ))}
            </div>
          )}
        </div>
      )}

      {/* Picture-in-Picture Source Footage Overlay (1 Video or 2 Videos Side-by-Side) */}
      {showSourcePip && effectiveSourceVideoUrl && (
        <div
          className={`absolute top-3 right-3 z-20 bg-[#0B0E17] border border-[#2D3B52] rounded-md overflow-hidden ${
            hasSecondaryVideo && effectiveSecondaryVideoUrl ? 'w-[420px]' : 'w-60'
          }`}
        >
          <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-[#1E293B] text-[11px] text-slate-300">
            <span className="font-medium">
              {hasSecondaryVideo && effectiveSecondaryVideoUrl
                ? 'Source Footage (View 1 & View 2)'
                : 'Source Footage'}
            </span>
            <button
              type="button"
              onClick={() => setShowSourcePip(false)}
              className="text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
          <div
            className={
              hasSecondaryVideo && effectiveSecondaryVideoUrl
                ? 'grid grid-cols-2 divide-x divide-[#1E293B]'
                : 'block'
            }
          >
            <div>
              {sourceMimeType?.startsWith('image/') ? (
                <img
                  src={effectiveSourceVideoUrl}
                  alt="Uploaded View 1 source frame"
                  referrerPolicy="no-referrer"
                  className="w-full h-36 object-cover"
                />
              ) : pipVideoError ? (
                <div className="w-full h-36 bg-[#07090E] p-3 flex flex-col items-center justify-center text-center gap-1">
                  <span className="text-[11px] font-medium text-amber-300">Codec Preview Unavailable</span>
                  <span className="text-[10px] text-slate-400">
                    View 1 is uploaded and ready for Gemini 4.
                  </span>
                </div>
              ) : (
                <video
                  key={effectiveSourceVideoUrl}
                  src={effectiveSourceVideoUrl}
                  controls
                  loop
                  muted
                  playsInline
                  preload="auto"
                  onError={() => setPipVideoError(true)}
                  className="w-full h-36 object-cover bg-black"
                />
              )}
            </div>

            {hasSecondaryVideo && effectiveSecondaryVideoUrl && (
              <div>
                {secondaryMimeType?.startsWith('image/') ? (
                  <img
                    src={effectiveSecondaryVideoUrl}
                    alt="Uploaded View 2 source frame"
                    referrerPolicy="no-referrer"
                    className="w-full h-36 object-cover"
                  />
                ) : pipSecondaryVideoError ? (
                  <div className="w-full h-36 bg-[#07090E] p-3 flex flex-col items-center justify-center text-center gap-1">
                    <span className="text-[11px] font-medium text-amber-300">Codec Preview Unavailable</span>
                    <span className="text-[10px] text-slate-400">
                      View 2 is uploaded and ready for Gemini 4.
                    </span>
                  </div>
                ) : (
                  <video
                    key={effectiveSecondaryVideoUrl}
                    src={effectiveSecondaryVideoUrl}
                    controls
                    loop
                    muted
                    playsInline
                    preload="auto"
                    onError={() => setPipSecondaryVideoError(true)}
                    className="w-full h-36 object-cover bg-black"
                  />
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Unified Collapsible Bottom HUD Dock (Zero Mid-Flank Clutter) */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 w-[min(95%,900px)] flex flex-col items-center">
        <button
          type="button"
          onClick={() => {
            logFunctionCall('toggleHudDock', { collapsed: !hudCollapsed });
            setHudCollapsed((v) => !v);
          }}
          className="mb-1 inline-flex items-center gap-1.5 px-3 py-1 text-[11px] font-medium text-slate-300 bg-[#111622]/95 border border-[#1E293B] rounded-md hover:text-white hover:border-[#2D3B52] transition-colors whitespace-nowrap cursor-pointer"
        >
          {hudCollapsed ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          <span>{hudCollapsed ? 'Show Controls' : 'Hide Controls'}</span>
        </button>

        {!hudCollapsed && (
          <div className="w-full bg-[#111622]/95 border border-[#1E293B] rounded-lg px-3 py-2 flex flex-col gap-2">
            {/* Optional Inline Live Physics Tuning Bar */}
            {showPhysicsPanel && (
              <div className="pb-2 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3 text-[11px]">
                <label className="flex items-center gap-2 text-slate-300">
                  <span className="font-mono-tabular text-slate-400">Friction μ</span>
                  <input
                    type="range"
                    min={0.14}
                    max={0.95}
                    step={0.02}
                    value={liveParams.friction}
                    onChange={(e) => {
                      const friction = Number(e.target.value);
                      if (simModeRef.current !== 'live') setSimMode('live');
                      setLiveParams((p) => ({ ...p, friction }));
                    }}
                    className="w-24 accent-cyan-400 cursor-pointer h-1.5"
                  />
                  <span className="font-mono-tabular text-cyan-300 w-8">
                    {liveParams.friction.toFixed(2)}
                  </span>
                </label>

                <label className="flex items-center gap-2 text-slate-300">
                  <span className="font-mono-tabular text-slate-400">Mass</span>
                  <input
                    type="range"
                    min={0.1}
                    max={1.1}
                    step={0.05}
                    value={liveParams.massKg}
                    onChange={(e) => {
                      const massKg = Number(e.target.value);
                      if (simModeRef.current !== 'live') setSimMode('live');
                      setLiveParams((p) => ({ ...p, massKg }));
                    }}
                    className="w-24 accent-amber-400 cursor-pointer h-1.5"
                  />
                  <span className="font-mono-tabular text-amber-300 w-12">
                    {liveParams.massKg.toFixed(2)} kg
                  </span>
                </label>

                <label className="flex items-center gap-2 text-slate-300">
                  <span className="font-mono-tabular text-slate-400">Clearance</span>
                  <input
                    type="range"
                    min={0.02}
                    max={0.18}
                    step={0.01}
                    value={liveParams.clearanceM}
                    onChange={(e) => {
                      const clearanceM = Number(e.target.value);
                      if (simModeRef.current !== 'live') setSimMode('live');
                      setLiveParams((p) => ({ ...p, clearanceM }));
                    }}
                    className="w-24 accent-emerald-400 cursor-pointer h-1.5"
                  />
                  <span className="font-mono-tabular text-emerald-300 w-10">
                    {Math.round(liveParams.clearanceM * 100)} cm
                  </span>
                </label>

                <button
                  type="button"
                  onClick={handleResetSandbox}
                  className="px-2.5 py-1 text-[11px] font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded hover:border-slate-400 transition-colors cursor-pointer"
                >
                  Reset Scene
                </button>
              </div>
            )}

            {/* Optional Inline 3D Video Clip Exporter Bar (15s / 30s / 60s 1-Minute WebM) */}
            {(showRecordPanel || isRecordingClip) && (
              <div className="pb-2 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-2 text-[11px]">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono-tabular text-slate-300 font-medium">
                    Export 3D Clip (.webm)
                  </span>
                  <div className="flex items-center gap-1 bg-[#0B0E17] p-0.5 rounded border border-[#1E293B]">
                    {APP_CONFIG.simulation.videoExport.presetsSec.map((sec) => (
                      <button
                        key={sec}
                        type="button"
                        disabled={isRecordingClip}
                        onClick={() => setRecordDurationSec(sec)}
                        className={`px-2 py-0.5 rounded font-mono-tabular transition-colors cursor-pointer ${
                          recordDurationSec === sec
                            ? 'bg-[#1E293B] text-amber-300'
                            : 'text-slate-400 hover:text-slate-200 disabled:opacity-50'
                        }`}
                      >
                        {sec === 60 ? '60s (1m)' : `${sec}s`}
                      </button>
                    ))}
                    {activeRollout?.durationSec &&
                      !APP_CONFIG.simulation.videoExport.presetsSec.includes(
                        Math.round(activeRollout.durationSec),
                      ) && (
                        <button
                          type="button"
                          disabled={isRecordingClip}
                          onClick={() =>
                            setRecordDurationSec(Math.round(activeRollout.durationSec ?? 15))
                          }
                          className={`px-2 py-0.5 rounded font-mono-tabular transition-colors cursor-pointer ${
                            recordDurationSec === Math.round(activeRollout.durationSec)
                              ? 'bg-[#1E293B] text-amber-300'
                              : 'text-slate-400 hover:text-slate-200 disabled:opacity-50'
                          }`}
                        >
                          Rollout ({Math.round(activeRollout.durationSec)}s)
                        </button>
                      )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {recordError && (
                    <span className="text-rose-400 text-[10px]">{recordError}</span>
                  )}
                  {isRecordingClip ? (
                    <>
                      <span className="font-mono-tabular text-rose-300 font-semibold">
                        REC {formatElapsedSeconds(recordElapsedSec)} /{' '}
                        {formatElapsedSeconds(recordDurationSec)}
                      </span>
                      <button
                        type="button"
                        onClick={handleStopClipRecording}
                        className="px-2.5 py-1 inline-flex items-center gap-1 font-semibold text-white bg-rose-600 rounded hover:bg-rose-500 transition-colors cursor-pointer"
                      >
                        <Square className="w-3 h-3" />
                        <span>Stop & Save</span>
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleStartClipRecording(recordDurationSec)}
                      className="px-2.5 py-1 inline-flex items-center gap-1 font-semibold text-slate-950 bg-amber-500 rounded hover:bg-amber-400 transition-colors cursor-pointer"
                    >
                      <Download className="w-3 h-3" />
                      <span>
                        Record {recordDurationSec === 60 ? '1-Min' : `${recordDurationSec}s`} Video
                      </span>
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Primary Unified Controls Row */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              {/* Playback Controls + Scrubber (Replay) OR Live Disturbance Controls */}
              <div className="flex items-center gap-1.5 min-w-[200px] flex-1">
                <button
                  type="button"
                  onClick={() => setIsPlaying((p) => !p)}
                  aria-label={isPlaying ? 'Pause simulation' : 'Play simulation'}
                  className="w-8 h-8 inline-flex items-center justify-center rounded-md bg-amber-500 text-slate-950 hover:bg-amber-400 transition-colors shrink-0 cursor-pointer"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>

                {effectiveSimMode === 'replay' ? (
                  <>
                    <button
                      type="button"
                      onClick={() => {
                        frameProgressRef.current = 0;
                        onFrameIndexChange(0);
                      }}
                      aria-label="Restart rollout"
                      className="w-8 h-8 inline-flex items-center justify-center rounded-md bg-[#161D2C] border border-[#1E293B] text-slate-300 hover:text-white transition-colors shrink-0 cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                    <input
                      type="range"
                      min={0}
                      max={Math.max(1, totalFrames - 1)}
                      value={currentFrameIndex}
                      onChange={(e) => {
                        const next = Number(e.target.value);
                        frameProgressRef.current = next;
                        onFrameIndexChange(next);
                      }}
                      aria-label="Rollout timeline scrubber"
                      className="w-full accent-amber-500 cursor-pointer h-1.5"
                    />
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={handleNudgeObject}
                      title="Apply a live horizontal impulse to the target object"
                      className="px-2.5 h-8 inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-200 bg-amber-500/15 border border-amber-500/40 rounded-md hover:bg-amber-500/25 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Wind className="w-3.5 h-3.5" />
                      <span>Nudge</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleResetSandbox}
                      title="Reset tabletop object positions"
                      className="w-8 h-8 inline-flex items-center justify-center rounded-md bg-[#161D2C] border border-[#1E293B] text-slate-300 hover:text-white transition-colors shrink-0 cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}

                {activeRollout && (
                  <div className="flex items-center gap-0.5 bg-[#0B0E17] p-0.5 rounded-md border border-[#1E293B] shrink-0">
                    <button
                      type="button"
                      onClick={() => {
                        logFunctionCall('RobotStage3D.selectSimMode', { mode: 'live' });
                        setSimMode('live');
                      }}
                      className={`px-2 py-1 text-[11px] font-medium rounded inline-flex items-center gap-1 transition-colors cursor-pointer ${
                        effectiveSimMode === 'live'
                          ? 'bg-cyan-500/20 text-cyan-300'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Activity className="w-3 h-3" />
                      <span>Live</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        logFunctionCall('RobotStage3D.selectSimMode', { mode: 'replay' });
                        setSimMode('replay');
                      }}
                      className={`px-2 py-1 text-[11px] font-medium rounded transition-colors cursor-pointer ${
                        effectiveSimMode === 'replay'
                          ? 'bg-[#1E293B] text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Replay
                    </button>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() =>
                    setPlaybackSpeed((s) => (s === 1 ? 2 : s === 2 ? 0.5 : 1))
                  }
                  className="px-2 py-1 text-xs font-mono-tabular text-slate-300 bg-[#161D2C] border border-[#1E293B] rounded hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
                >
                  {playbackSpeed}x
                </button>
              </div>

              {/* Optional Dual Viewport Split Mode — Only shown when 2 videos are uploaded */}
              {hasSecondaryVideo && (
                <div className="flex items-center gap-0.5 bg-[#0B0E17] p-0.5 rounded-md border border-[#1E293B]">
                  {(Object.keys(APP_CONFIG.stageViewModes) as StageViewModeId[])
                    .filter((layoutKey) => !(hideRobotArm && layoutKey === 'dual_arms'))
                    .map((layoutKey) => (
                      <button
                        key={layoutKey}
                        type="button"
                        title={APP_CONFIG.stageViewModes[layoutKey].description}
                        onClick={() => {
                          logFunctionCall('RobotStage3D.selectStageLayout', { layoutKey });
                          setStageLayout(layoutKey);
                        }}
                        className={`px-2 py-1 text-[11px] font-medium rounded inline-flex items-center gap-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          effectiveStageLayout === layoutKey
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        {layoutKey === 'single' ? (
                          <Square className="w-3 h-3" />
                        ) : (
                          <Columns2 className="w-3 h-3" />
                        )}
                        <span>{APP_CONFIG.stageViewModes[layoutKey].label}</span>
                      </button>
                    ))}
                </div>
              )}

              {/* Camera Presets */}
              <div className="flex items-center gap-1 bg-[#0B0E17] p-0.5 rounded-md border border-[#1E293B]">
                {(Object.keys(APP_CONFIG.cameraPresets) as CameraPresetId[]).map((presetKey) => (
                  <button
                    key={presetKey}
                    type="button"
                    onClick={() => handleSelectCameraPreset(presetKey)}
                    className={`px-2 py-1 text-[11px] font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      cameraPreset === presetKey
                        ? 'bg-[#1E293B] text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {presetKey === 'perspective'
                      ? '3D'
                      : presetKey === 'top'
                        ? 'Top'
                        : presetKey === 'gripper'
                          ? 'Gripper'
                          : 'Side'}
                  </button>
                ))}
              </div>

              {/* Layer, Live Physics Drawer, Record Clip, Expand & Media Toggles */}
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    if (isRecordingClip) {
                      handleStopClipRecording();
                    } else {
                      logFunctionCall('RobotStage3D.toggleRecordPanel', {
                        open: !showRecordPanel,
                      });
                      setShowRecordPanel((v) => !v);
                    }
                  }}
                  title={
                    isRecordingClip
                      ? 'Stop recording and download WebM clip'
                      : 'Record 15s, 30s, or 60s (1-minute) 3D simulation clip'
                  }
                  className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                    isRecordingClip
                      ? 'bg-rose-500/20 border-rose-500/60 text-rose-200 font-mono-tabular font-semibold'
                      : showRecordPanel
                        ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                        : 'bg-[#161D2C] border-[#1E293B] text-slate-300 hover:text-white'
                  }`}
                >
                  <Circle
                    className={`w-3 h-3 ${isRecordingClip ? 'fill-rose-400 text-rose-400' : 'text-rose-400'}`}
                  />
                  <span>
                    {isRecordingClip
                      ? `REC ${formatElapsedSeconds(recordElapsedSec)}`
                      : 'Record'}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    logFunctionCall('RobotStage3D.togglePhysicsPanel', {
                      open: !showPhysicsPanel,
                    });
                    setShowPhysicsPanel((v) => !v);
                  }}
                  title="Live friction, mass, and clearance tuning"
                  className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                    showPhysicsPanel
                      ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-200'
                      : 'bg-[#161D2C] border-[#1E293B] text-slate-300 hover:text-white'
                  }`}
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  <span>Physics</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowTrajectory((v) => !v)}
                  title="Toggle trajectory spline"
                  className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                    showTrajectory
                      ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                      : 'bg-[#161D2C] border-[#1E293B] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Route className="w-3.5 h-3.5" />
                  <span>Path</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowForceVector((v) => !v)}
                  title="Toggle contact force vector"
                  className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                    showForceVector
                      ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                      : 'bg-[#161D2C] border-[#1E293B] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Force</span>
                </button>
                <button
                  type="button"
                  onClick={() => setShowGrid((v) => !v)}
                  title="Toggle metric grid"
                  className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                    showGrid
                      ? 'bg-slate-700/40 border-slate-500/50 text-slate-200'
                      : 'bg-[#161D2C] border-[#1E293B] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Grid3x3 className="w-3.5 h-3.5" />
                </button>
                {effectiveSourceVideoUrl && (
                  <button
                    type="button"
                    onClick={() => setShowSourcePip((v) => !v)}
                    title="Compare uploaded source footage"
                    className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                      showSourcePip
                        ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                        : 'bg-[#161D2C] border-[#1E293B] text-slate-300 hover:text-white'
                    }`}
                  >
                    <Video className="w-3.5 h-3.5" />
                    <span>PiP</span>
                  </button>
                )}
                {activeRollout?.previewVideoUrl && (
                  <button
                    type="button"
                    onClick={() => setViewMode((m) => (m === '3d' ? 'cloud_mp4' : '3d'))}
                    title="Switch between Three.js 3D stage and Cloud MuJoCo MP4"
                    className={`px-2 py-1 text-[11px] inline-flex items-center gap-1 rounded border transition-colors whitespace-nowrap cursor-pointer ${
                      viewMode === 'cloud_mp4'
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-200'
                        : 'bg-[#161D2C] border-[#1E293B] text-slate-300 hover:text-white'
                    }`}
                  >
                    {viewMode === 'cloud_mp4' ? (
                      <>
                        <Camera className="w-3.5 h-3.5" />
                        <span>3D Sim</span>
                      </>
                    ) : (
                      <>
                        <Film className="w-3.5 h-3.5" />
                        <span>Cloud MP4</span>
                      </>
                    )}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    logFunctionCall('RobotStage3D.toggleExpandedStage', {
                      expanded: !expandedStage,
                    });
                    setExpandedStage((v) => !v);
                  }}
                  title={expandedStage ? 'Compact stage height' : 'Expand stage height'}
                  aria-label={expandedStage ? 'Compact stage height' : 'Expand stage height'}
                  className="px-2 py-1 text-[11px] inline-flex items-center justify-center rounded border bg-[#161D2C] border-[#1E293B] text-slate-300 hover:text-white transition-colors cursor-pointer"
                >
                  {expandedStage ? (
                    <Minimize2 className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize2 className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
