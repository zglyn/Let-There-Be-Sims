/**
 * Sim2Real Compiler — Root Application Orchestrator (`App.tsx`)
 *
 * Coordinates both runtime modes of the Sim-to-Real Robotics Failure-to-Policy
 * Compiler:
 * 1. Standalone Mobile Capture Route (`/capture?session=<id>`) when scanned
 *    from a smartphone camera via QR code.
 * 2. Desktop Precision Lab Dark Command Center (`/`) featuring the strict
 *    3-zone `TopBar`, asymmetric split console (`CaptureIntakePanel` +
 *    `PipelineRail` on the left; persistent `RobotStage3D` + `SwarmMatrix` +
 *    `TelemetryChart` on the main stage), and contextual views (`Mobile Capture`,
 *    `Policy Diff`, `Cloud Swarm`, `Bundle`) that never tear down the active
 *    SSE stream or WebGL session state.
 *
 * Use cases:
 * - End-to-end phone-to-cloud-sim robotics failure diagnosis and policy repair.
 * - Live streaming SSE state updates across Carbon, Hydrogen, and Neon turns.
 * - Automatic background polling of Long Decode deployment bundles.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { APP_CONFIG, type RobotModelId } from './config/appConfig';
import { BundleExportView } from './components/BundleExportView';
import {
  CaptureIntakePanel,
  type LocalVideoMeta,
  type RolloutDurationSelection,
} from './components/CaptureIntakePanel';
import { CloudSwarmView } from './components/CloudSwarmView';
import { MobileCaptureView } from './components/MobileCaptureView';
import { PipelineRail } from './components/PipelineRail';
import { PolicyDiffView } from './components/PolicyDiffView';
import { RobotStage3D } from './components/RobotStage3D';
import { SessionExportBar } from './components/SessionExportBar';
import { SpatialSceneMap } from './components/SpatialSceneMap';
import { SwarmMatrix } from './components/SwarmMatrix';
import { TelemetryChart } from './components/TelemetryChart';
import { TopBar, type ActiveWorkspaceView } from './components/TopBar';
import {
  cancelPipelineRun,
  createCompilerSession,
  fetchCompilerSession,
  openSessionEventStream,
  pollDeploymentBundle,
  removeSessionVideo,
  saveWebhookConfig,
  testWebhookConnection,
  triggerPipelineRun,
  uploadSessionVideo,
} from './services/compilerApi';
import type { CompilerSessionState, ServerStreamEvent, VideoSlot } from './types/robotics';
import { logFunctionCall } from './utils/logger';

/**
 * Parses URL pathname and `?session=` query parameter to detect standalone
 * mobile QR capture visits vs. desktop command center sessions.
 */
function parseInitialRoute(): { isMobileCaptureRoute: boolean; querySessionId: string | null } {
  if (typeof window === 'undefined') {
    return { isMobileCaptureRoute: false, querySessionId: null };
  }
  const url = new URL(window.location.href);
  return {
    isMobileCaptureRoute: url.pathname.startsWith('/capture'),
    querySessionId: url.searchParams.get('session'),
  };
}

/**
 * Root React component for the Sim2Real Compiler desktop and mobile experiences.
 */
export default function App() {
  const [{ isMobileCaptureRoute, querySessionId }] = useState(parseInitialRoute);

  const [session, setSession] = useState<CompilerSessionState | null>(null);
  const [activeView, setActiveView] = useState<ActiveWorkspaceView>('command_center');
  const [taskGoal, setTaskGoal] = useState<string>(APP_CONFIG.simulation.defaultTaskGoal);
  const [robotModel, setRobotModel] = useState<RobotModelId>(APP_CONFIG.simulation.defaultRobotModel);
  const [durationMode, setDurationMode] = useState<RolloutDurationSelection>('match');
  const [detectedVideoDurationSec, setDetectedVideoDurationSec] = useState<number | null>(null);
  const [selectedIterationIndex, setSelectedIterationIndex] = useState(0);
  const [selectedEnvId, setSelectedEnvId] = useState(1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [lowerPanelTab, setLowerPanelTab] = useState<'swarm' | 'spatial_map'>('swarm');
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadStatusLabel, setUploadStatusLabel] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [localVideoPreviewUrl, setLocalVideoPreviewUrl] = useState<string | null>(null);
  const [localVideoMeta, setLocalVideoMeta] = useState<LocalVideoMeta | null>(null);
  const [localSecondaryVideoPreviewUrl, setLocalSecondaryVideoPreviewUrl] = useState<string | null>(
    null,
  );
  const [localSecondaryVideoMeta, setLocalSecondaryVideoMeta] = useState<LocalVideoMeta | null>(
    null,
  );
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [isRefreshingBundle, setIsRefreshingBundle] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  const localPreviewUrlRef = useRef<string | null>(null);
  const localSecondaryPreviewUrlRef = useRef<string | null>(null);

  /**
   * Replaces the local blob preview URL for View 1 (primary) while revoking
   * any prior object URL to prevent browser memory leaks.
   */
  const replaceLocalPreview = useCallback(
    (nextUrl: string | null, meta: LocalVideoMeta | null) => {
      if (localPreviewUrlRef.current && localPreviewUrlRef.current !== nextUrl) {
        URL.revokeObjectURL(localPreviewUrlRef.current);
      }
      localPreviewUrlRef.current = nextUrl;
      setLocalVideoPreviewUrl(nextUrl);
      setLocalVideoMeta(meta);
    },
    [],
  );

  /**
   * Replaces the local blob preview URL for optional View 2 (secondary) while
   * revoking any prior object URL to prevent browser memory leaks.
   */
  const replaceLocalSecondaryPreview = useCallback(
    (nextUrl: string | null, meta: LocalVideoMeta | null) => {
      if (
        localSecondaryPreviewUrlRef.current &&
        localSecondaryPreviewUrlRef.current !== nextUrl
      ) {
        URL.revokeObjectURL(localSecondaryPreviewUrlRef.current);
      }
      localSecondaryPreviewUrlRef.current = nextUrl;
      setLocalSecondaryVideoPreviewUrl(nextUrl);
      setLocalSecondaryVideoMeta(meta);
    },
    [],
  );

  // Revoke any active object URLs on unmount
  useEffect(() => {
    return () => {
      if (localPreviewUrlRef.current) {
        URL.revokeObjectURL(localPreviewUrlRef.current);
        localPreviewUrlRef.current = null;
      }
      if (localSecondaryPreviewUrlRef.current) {
        URL.revokeObjectURL(localSecondaryPreviewUrlRef.current);
        localSecondaryPreviewUrlRef.current = null;
      }
    };
  }, []);

  /**
   * Displays a concise non-blocking confirmation toast for 3.5 seconds.
   */
  const showToast = useCallback((message: string) => {
    logFunctionCall('App.showToast', { message });
    setToastMessage(message);
    window.setTimeout(() => {
      setToastMessage((prev) => (prev === message ? null : prev));
    }, 3500);
  }, []);

  /**
   * Initializes a fresh desktop session or hydrates an existing one from URL.
   */
  const initializeSession = useCallback(async (existingId?: string | null) => {
    logFunctionCall('App.initializeSession', { existingId: existingId ?? null });
    setIsCreatingSession(true);
    setInitError(null);
    try {
      let nextSession: CompilerSessionState;
      if (existingId) {
        try {
          nextSession = await fetchCompilerSession(existingId);
        } catch {
          nextSession = await createCompilerSession();
        }
      } else {
        nextSession = await createCompilerSession();
      }
      setSession(nextSession);
      setTaskGoal(nextSession.taskGoal);
      setRobotModel(nextSession.robotModel);
      setSelectedIterationIndex(Math.max(0, nextSession.iterations.length - 1));
      setSelectedEnvId(1);
      setCurrentFrameIndex(0);
      setDetectedVideoDurationSec(null);
      setUploadError(null);
      replaceLocalPreview(null, null);
      replaceLocalSecondaryPreview(null, null);
      setActiveView('command_center');
    } catch (err) {
      setInitError(
        err instanceof Error ? err.message : 'Unable to connect to LetThereBeSims backend.',
      );
    } finally {
      setIsCreatingSession(false);
    }
  }, [replaceLocalPreview, replaceLocalSecondaryPreview]);

  // Boot desktop session once (skip if rendering standalone mobile capture route)
  useEffect(() => {
    if (isMobileCaptureRoute) return;
    void initializeSession(querySessionId);
  }, [isMobileCaptureRoute, querySessionId, initializeSession]);

  // Subscribe to real-time SSE events for the active desktop session
  useEffect(() => {
    if (isMobileCaptureRoute || !session?.id) return;

    const handleSseEvent = (event: ServerStreamEvent) => {
      switch (event.type) {
        case 'session_snapshot':
          setSession(event.session);
          break;
        case 'video_uploaded':
          setSession((prev) =>
            prev
              ? {
                  ...prev,
                  video: event.slot === 'primary' ? event.video : prev.video,
                  secondaryVideo:
                    event.slot === 'secondary' ? event.video : prev.secondaryVideo,
                  status: event.sessionStatus,
                  error: null,
                }
              : prev,
          );
          setUploadError(null);
          if (event.video.source === 'mobile_qr') {
            if (event.slot === 'secondary') {
              replaceLocalSecondaryPreview(null, null);
            } else {
              replaceLocalPreview(null, null);
            }
            showToast(`Smartphone footage received (${event.video.originalName}).`);
          }
          break;
        case 'video_removed':
          setSession((prev) =>
            prev
              ? {
                  ...prev,
                  video: event.slot === 'primary' ? null : prev.video,
                  secondaryVideo:
                    event.slot === 'secondary' ? null : prev.secondaryVideo,
                  status: event.sessionStatus,
                }
              : prev,
          );
          if (event.slot === 'secondary') {
            replaceLocalSecondaryPreview(null, null);
          } else {
            replaceLocalPreview(null, null);
          }
          break;
        case 'stage_update':
          setSession((prev) =>
            prev
              ? {
                  ...prev,
                  stages: event.stages,
                  status: event.status,
                  progress: event.progress ?? prev.progress,
                }
              : prev,
          );
          break;
        case 'trace_appended':
          setSession((prev) =>
            prev
              ? { ...prev, interactionTrace: [...prev.interactionTrace, event.entry] }
              : prev,
          );
          break;
        case 'scene_ready':
          setSession((prev) => (prev ? { ...prev, sceneSpec: event.sceneSpec } : prev));
          if (event.sceneSpec.sceneMode === 'scene_map_only') {
            setLowerPanelTab('spatial_map');
            showToast('No robot detected — simulation-ready 3D scene & 2D spatial map compiled.');
          } else if (event.sceneSpec.sceneMode === 'robot_success_verification') {
            setLowerPanelTab('swarm');
            showToast('Successful demo detected — running 20-environment robustness stress test.');
          } else {
            setLowerPanelTab('swarm');
            showToast('Failure detected — Gemini 4 Carbon compiled MuJoCo repair scene.');
          }
          break;
        case 'iteration_complete':
          setSession((prev) => {
            if (!prev) return prev;
            const nextIters = [...prev.iterations, event.iteration];
            setSelectedIterationIndex(nextIters.length - 1);
            // Focus on first failing env if any exist, otherwise Env 1
            const firstFail = event.iteration.rollouts.find((r) => !r.success);
            setSelectedEnvId(firstFail ? firstFail.envId : 1);
            setCurrentFrameIndex(0);
            return { ...prev, iterations: nextIters };
          });
          break;
        case 'bundle_update':
          setSession((prev) => (prev ? { ...prev, bundle: event.bundle } : prev));
          break;
        case 'pipeline_error':
          setSession((prev) =>
            prev ? { ...prev, status: 'error', error: event.message } : prev,
          );
          break;
        case 'pipeline_complete':
          setSession(event.session);
          showToast('Closed-loop policy verification complete.');
          break;
      }
    };

    const cleanup = openSessionEventStream(session.id, handleSseEvent);
    return cleanup;
  }, [
    isMobileCaptureRoute,
    session?.id,
    showToast,
    replaceLocalPreview,
    replaceLocalSecondaryPreview,
  ]);

  // Automatically poll background Long Decode bundle while in_progress
  useEffect(() => {
    if (!session?.id || session.bundle?.status !== 'in_progress') return;
    const interval = window.setInterval(async () => {
      try {
        const updated = await pollDeploymentBundle(session.id);
        if (updated) {
          setSession((prev) => (prev ? { ...prev, bundle: updated } : prev));
          if (updated.status === 'completed') {
            showToast('Long Decode deployment bundle is ready for download.');
          }
        }
      } catch {
        // Keep silent during background polling; manual refresh surfaces errors
      }
    }, 6000);
    return () => window.clearInterval(interval);
  }, [session?.id, session?.bundle?.status, showToast]);

  // Standalone Mobile QR Capture Route
  if (isMobileCaptureRoute) {
    if (!querySessionId) {
      return (
        <div className="min-h-screen bg-[#07090E] text-slate-100 flex items-center justify-center p-6 text-center">
          <div className="max-w-sm bg-[#111622] border border-[#1E293B] rounded-lg p-6">
            <h1 className="font-display text-lg font-bold">Missing Pairing Session</h1>
            <p className="mt-2 text-xs text-slate-400">
              Scan the QR code displayed on the LetThereBeSims desktop command center to link your smartphone camera.
            </p>
          </div>
        </div>
      );
    }
    return <MobileCaptureView sessionId={querySessionId} standalone />;
  }

  /**
   * Handles uploading 1 video (Single View) or 2 videos simultaneously (Dual View)
   * from desktop file selection, drag-and-drop, or webcam capture.
   */
  const handleDesktopUploadFiles = async (
    items: Array<{ file: File | Blob; filename: string; slot?: VideoSlot }>,
    source: 'desktop_upload' | 'desktop_webcam',
  ): Promise<void> => {
    if (!session || items.length === 0) return;
    logFunctionCall('App.handleDesktopUploadFiles', {
      sessionId: session.id,
      count: items.length,
      slots: items.map((i) => i.slot ?? 'primary'),
      source,
    });
    setUploadError(null);

    const uploadingOnlyPrimary =
      items.length === 1 && (items[0].slot ?? 'primary') === 'primary';

    // Immediately show local previews for all chosen videos
    for (const item of items) {
      const slot: VideoSlot = item.slot === 'secondary' ? 'secondary' : 'primary';
      const objectUrl = URL.createObjectURL(item.file);
      const meta: LocalVideoMeta = {
        name: item.filename,
        mimeType: item.file.type || 'video/mp4',
        sizeBytes: item.file.size,
      };
      if (slot === 'secondary') {
        replaceLocalSecondaryPreview(objectUrl, meta);
      } else {
        replaceLocalPreview(objectUrl, meta);
      }
    }

    setIsUploading(true);
    setUploadProgress(0);
    try {
      // If user explicitly uploaded a single new primary video while an old View 2 existed,
      // clear View 2 so a 1-video upload cleanly yields Single View.
      if (uploadingOnlyPrimary && (session.secondaryVideo || localSecondaryVideoPreviewUrl)) {
        replaceLocalSecondaryPreview(null, null);
        try {
          const cleared = await removeSessionVideo(session.id, 'secondary');
          setSession(cleared);
        } catch {
          // Non-fatal if secondary slot was already empty on server
        }
      }

      let latestSession = session;
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const slot: VideoSlot = item.slot === 'secondary' ? 'secondary' : 'primary';
        setUploadStatusLabel(
          items.length > 1
            ? `Uploading View ${i + 1} of ${items.length}...`
            : slot === 'secondary'
              ? 'Uploading View 2...'
              : 'Uploading View 1...',
        );
        const result = await uploadSessionVideo({
          sessionId: session.id,
          slot,
          file: item.file,
          filename: item.filename,
          source,
          taskGoal,
          onProgress: (pct) => {
            const combined = Math.round(((i + pct / 100) / items.length) * 100);
            setUploadProgress(combined);
          },
        });
        latestSession = result.session;
        setSession(result.session);
      }

      const dualActive = Boolean(latestSession.secondaryVideo);
      showToast(
        dualActive
          ? '2 camera views staged — Dual View unlocked. Click Analyze & Verify Policy.'
          : '1 video staged (Single View). Click Analyze & Verify Policy.',
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed.';
      setUploadError(msg);
      for (const item of items) {
        if (item.slot === 'secondary') {
          replaceLocalSecondaryPreview(null, null);
        } else {
          replaceLocalPreview(null, null);
        }
      }
      setSession((prev) => (prev ? { ...prev, error: msg } : prev));
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
      setUploadStatusLabel(null);
    }
  };

  /**
   * Removes the optional secondary video (View 2) and returns the 3D stage to Single View.
   */
  const handleRemoveSecondaryVideo = async (): Promise<void> => {
    if (!session) return;
    logFunctionCall('App.handleRemoveSecondaryVideo', { sessionId: session.id });
    replaceLocalSecondaryPreview(null, null);
    try {
      const updated = await removeSessionVideo(session.id, 'secondary');
      setSession(updated);
      showToast('Removed View 2 — returned to Single View.');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unable to remove View 2.';
      setUploadError(msg);
    }
  };

  const effectiveRolloutDurationSec =
    durationMode === 'match'
      ? detectedVideoDurationSec
        ? Math.max(
            APP_CONFIG.simulation.minRolloutDurationSec,
            Math.min(APP_CONFIG.simulation.maxRolloutDurationSec, Math.round(detectedVideoDurationSec)),
          )
        : APP_CONFIG.simulation.defaultRolloutDurationSec
      : durationMode;

  /**
   * Triggers the Gemini 4 pipeline on the backend. When `forceSimulateRobot`
   * is true, mounts the selected robot arm onto an already-mapped robot-free
   * scene and executes Stages 2–4 (20-Env Swarm, Hydrogen, Neon, Long Decode).
   */
  const handleRunPipeline = async (forceSimulateRobot = false): Promise<void> => {
    if (!session) return;
    logFunctionCall('App.handleRunPipeline', {
      sessionId: session.id,
      taskGoal,
      robotModel,
      rolloutDurationSec: effectiveRolloutDurationSec,
      forceSimulateRobot,
    });
    try {
      const nowIso = new Date().toISOString();
      const progCfg = APP_CONFIG.pipelineProgress;
      setSession((prev) =>
        prev
          ? {
              ...prev,
              status: 'running',
              error: null,
              rolloutDurationSec: effectiveRolloutDurationSec,
              progress: {
                percent: forceSimulateRobot
                  ? progCfg.carbonTargetPct
                  : progCfg.mediaPrepStartPct,
                targetPercent: forceSimulateRobot
                  ? progCfg.swarmTargetPct
                  : progCfg.mediaPrepTargetPct,
                expectedStageDurationSec: forceSimulateRobot
                  ? progCfg.swarmExpectedSec
                  : progCfg.mediaPrepExpectedSec,
                stageLabel: forceSimulateRobot
                  ? 'Stage 2/4 · Mounting Robot & Running 20-Env Swarm'
                  : 'Stage 1/4 · Preparing Video Footage',
                stageUpdatedAt: nowIso,
                startedAt: nowIso,
              },
            }
          : prev,
      );
      if (forceSimulateRobot) {
        setLowerPanelTab('swarm');
      }
      await triggerPipelineRun({
        sessionId: session.id,
        taskGoal,
        robotModel,
        rolloutDurationSec: effectiveRolloutDurationSec,
        forceSimulateRobot,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to launch pipeline.';
      setSession((prev) => (prev ? { ...prev, status: 'error', error: msg } : prev));
    }
  };

  /**
   * Cancels an active Gemini 4 pipeline run immediately so long or stalled runs
   * can be stopped cleanly without refreshing the page.
   */
  const handleCancelPipeline = async (): Promise<void> => {
    if (!session) return;
    logFunctionCall('App.handleCancelPipeline', { sessionId: session.id });
    try {
      const updated = await cancelPipelineRun(session.id);
      setSession(updated);
      showToast('Pipeline run cancelled.');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to cancel pipeline.';
      setSession((prev) => (prev ? { ...prev, status: 'error', error: msg } : prev));
    }
  };

  /**
   * Saves external Cloud MuJoCo webhook settings on the active session.
   */
  const handleSaveWebhook = async (params: {
    enabled: boolean;
    url: string;
    authToken?: string;
  }): Promise<void> => {
    if (!session) return;
    const webhook = await saveWebhookConfig({
      sessionId: session.id,
      ...params,
    });
    setSession((prev) => (prev ? { ...prev, webhook } : prev));
  };

  /**
   * Manually refreshes the background Long Decode deployment bundle status.
   */
  const handleRefreshBundle = async (): Promise<void> => {
    if (!session) return;
    setIsRefreshingBundle(true);
    try {
      const bundle = await pollDeploymentBundle(session.id);
      if (bundle) {
        setSession((prev) => (prev ? { ...prev, bundle } : prev));
      }
    } finally {
      setIsRefreshingBundle(false);
    }
  };

  const activeIteration = session?.iterations[selectedIterationIndex] ?? null;
  const activeRollout =
    activeIteration?.rollouts.find((r) => r.envId === selectedEnvId) ??
    activeIteration?.rollouts[0] ??
    null;

  return (
    <div className="min-h-screen bg-[#07090E] text-[#E6ECF5] flex flex-col">
      <TopBar
        activeView={activeView}
        onSelectView={setActiveView}
        onCreateNewSession={() => void initializeSession(null)}
        isCreatingSession={isCreatingSession}
      />

      {/* Non-blocking Toast Banner */}
      {toastMessage && (
        <div
          role="status"
          className="fixed bottom-5 right-5 z-50 bg-[#161D2C] border border-amber-500/50 text-slate-100 text-xs px-4 py-2.5 rounded-md shadow-lg"
        >
          {toastMessage}
        </div>
      )}

      <main className="flex-1 w-full max-w-[1600px] mx-auto px-5 py-5">
        {initError ? (
          <div className="max-w-lg mx-auto mt-16 bg-[#111622] border border-rose-500/40 rounded-lg p-6 text-center">
            <h1 className="font-display text-lg font-semibold text-rose-300">
              Backend Connection Unavailable
            </h1>
            <p className="mt-2 text-xs text-slate-300">{initError}</p>
            <button
              type="button"
              onClick={() => void initializeSession(null)}
              className="mt-4 px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 rounded hover:bg-amber-400 cursor-pointer"
            >
              Retry Connection
            </button>
          </div>
        ) : !session ? (
          <div className="py-24 text-center text-xs font-mono-tabular text-slate-400">
            Initializing LetThereBeSims session...
          </div>
        ) : (
          <>
            {/* Primary Single-Screen Command Center (kept mounted to preserve Three.js WebGL context) */}
            <div
              className={
                activeView === 'command_center'
                  ? 'grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-5 items-start'
                  : 'hidden'
              }
            >
              {/* Left Rail: QR/Video Intake + Live 4-Stage Gemini 4 Pipeline */}
              <div className="flex flex-col gap-5">
                <CaptureIntakePanel
                  session={session}
                  taskGoal={taskGoal}
                  robotModel={robotModel}
                  durationMode={durationMode}
                  detectedVideoDurationSec={detectedVideoDurationSec}
                  uploadProgress={uploadProgress}
                  uploadStatusLabel={uploadStatusLabel}
                  isUploading={isUploading}
                  localVideoPreviewUrl={localVideoPreviewUrl}
                  localVideoMeta={localVideoMeta}
                  localSecondaryVideoPreviewUrl={localSecondaryVideoPreviewUrl}
                  localSecondaryVideoMeta={localSecondaryVideoMeta}
                  uploadError={uploadError}
                  onTaskGoalChange={setTaskGoal}
                  onRobotModelChange={setRobotModel}
                  onDurationModeChange={setDurationMode}
                  onDetectedVideoDuration={setDetectedVideoDurationSec}
                  onUploadFiles={handleDesktopUploadFiles}
                  onRemoveSecondaryVideo={handleRemoveSecondaryVideo}
                  onRunPipeline={handleRunPipeline}
                  onCancelPipeline={handleCancelPipeline}
                  onOpenMobilePreview={() => setActiveView('mobile_capture')}
                />
                <PipelineRail
                  sessionId={session.id}
                  sessionStatus={session.status}
                  progress={session.progress}
                  stages={session.stages}
                  interactionTrace={session.interactionTrace}
                  webhookEnabled={session.webhook.enabled}
                  error={session.error}
                />
              </div>

              {/* Center Stage: Export Bar + Interactive 3D Robot Stage + 20-Env Matrix + Dual-Axis Telemetry */}
              <div className="flex flex-col gap-5 min-w-0">
                <SessionExportBar
                  session={session}
                  robotModel={robotModel}
                  onExported={showToast}
                />

                <RobotStage3D
                  robotModel={robotModel}
                  sceneSpec={session.sceneSpec}
                  activeRollout={activeRollout}
                  sourceVideoUrl={session.video?.mediaUrl ?? null}
                  localVideoPreviewUrl={localVideoPreviewUrl}
                  sourceMimeType={session.video?.mimeType ?? localVideoMeta?.mimeType ?? null}
                  hasSecondaryVideo={Boolean(
                    session.secondaryVideo || localSecondaryVideoPreviewUrl,
                  )}
                  secondaryVideoUrl={session.secondaryVideo?.mediaUrl ?? null}
                  localSecondaryVideoPreviewUrl={localSecondaryVideoPreviewUrl}
                  secondaryMimeType={
                    session.secondaryVideo?.mimeType ??
                    localSecondaryVideoMeta?.mimeType ??
                    null
                  }
                  iterationNumber={activeIteration?.iteration ?? null}
                  currentFrameIndex={currentFrameIndex}
                  hideRobotArm={session.sceneSpec?.sceneMode === 'scene_map_only'}
                  onFrameIndexChange={setCurrentFrameIndex}
                  onClipExported={showToast}
                />

                {/* Lower Stage Mode Switcher: 2D Spatial Map vs 20-Env Swarm & Telemetry */}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-1 p-1 bg-[#111622] border border-[#1E293B] rounded-md">
                    <button
                      type="button"
                      onClick={() => setLowerPanelTab('spatial_map')}
                      className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                        lowerPanelTab === 'spatial_map'
                          ? 'bg-[#1E293B] text-white'
                          : 'text-slate-400 hover:text-slate-100'
                      }`}
                    >
                      2D Spatial Map
                    </button>
                    <button
                      type="button"
                      onClick={() => setLowerPanelTab('swarm')}
                      className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                        lowerPanelTab === 'swarm'
                          ? 'bg-[#1E293B] text-white'
                          : 'text-slate-400 hover:text-slate-100'
                      }`}
                    >
                      20-Env Swarm & Telemetry
                    </button>
                  </div>

                  {session.sceneSpec?.sceneMode === 'scene_map_only' && (
                    <span className="font-mono-tabular text-xs text-cyan-400">
                      Simulation-Ready Scene · No Robot in Source Footage
                    </span>
                  )}
                </div>

                {lowerPanelTab === 'spatial_map' ? (
                  <SpatialSceneMap
                    sessionId={session.id}
                    sceneSpec={session.sceneSpec}
                    robotModel={robotModel}
                    isPipelineRunning={session.status === 'running'}
                    onMountAndSimulate={() => handleRunPipeline(true)}
                    onExported={showToast}
                  />
                ) : (
                  <>
                    <SwarmMatrix
                      iterations={session.iterations}
                      selectedIterationIndex={selectedIterationIndex}
                      selectedEnvId={selectedEnvId}
                      onSelectIteration={(idx) => {
                        setSelectedIterationIndex(idx);
                        setCurrentFrameIndex(0);
                      }}
                      onSelectEnv={(envId) => {
                        setSelectedEnvId(envId);
                        setCurrentFrameIndex(0);
                      }}
                    />

                    <TelemetryChart
                      rollout={activeRollout}
                      currentFrameIndex={currentFrameIndex}
                      onSeekFrame={setCurrentFrameIndex}
                    />
                  </>
                )}
              </div>
            </div>

            {/* Contextual View: Mobile Capture Preview */}
            {activeView === 'mobile_capture' && (
              <div className="py-2">
                <MobileCaptureView
                  sessionId={session.id}
                  standalone={false}
                  onUploadedToDesktop={(updated) => {
                    setSession(updated);
                    setActiveView('command_center');
                  }}
                />
              </div>
            )}

            {/* Contextual View: Policy Diff & MJCF XML */}
            {activeView === 'policy_diff' && (
              <PolicyDiffView
                sceneSpec={session.sceneSpec}
                iterations={session.iterations}
                selectedIterationIndex={selectedIterationIndex}
                onSelectIteration={setSelectedIterationIndex}
              />
            )}

            {/* Contextual View: External Cloud MuJoCo Swarm Connector */}
            {activeView === 'cloud_swarm' && (
              <CloudSwarmView
                webhook={session.webhook}
                onSaveWebhook={handleSaveWebhook}
                onTestWebhook={testWebhookConnection}
              />
            )}

            {/* Contextual View: Long Decode Deployment Bundle */}
            {activeView === 'bundle' && (
              <BundleExportView
                bundle={session.bundle}
                onRefreshBundle={handleRefreshBundle}
                isRefreshing={isRefreshingBundle}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
