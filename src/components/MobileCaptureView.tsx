/**
 * Mobile-First QR-Paired Judge Hand-Demo & Camera Capture Route (`MobileCaptureView`)
 *
 * Dedicated touch-ergonomic capture interface served at `/capture?session=<id>`
 * when a judge scans the desktop QR code (and also previewable inline from the
 * desktop Top Bar). Implements the complete 3-step live demonstration capture:
 * 1. **Tap-to-Select Target Object:** Tapping directly on the live rear-camera
 *    viewfinder drops a precision crosshair reticle and records normalized
 *    `(tapX, tapY)` coordinates anchored to the physical tabletop object.
 * 2. **6-Second Hand Demo + DeviceOrientation Gravity Vector + Audio Track:**
 *    Captures rear-camera video + live microphone audio (`audio: true` with RMS
 *    meter) + live `DeviceOrientation` / `DeviceMotion` gravity vector
 *    `(gx, gy, gz)` & camera tilt degrees, with an automatic 6.0-second
 *    countdown that stops and streams immediately to the paired desktop session.
 * 3. **Automatic SO-ARM101 Retargeting Launch:** Streams video + audio + gravity
 *    vector + tap coordinates to the backend with `autoStart: true`, triggering
 *    Gemini 4 Carbon (MuJoCo scene + hand-to-SO-ARM101 trajectory retargeting),
 *    Hydrogen (20 randomized rollouts), and Neon (slip/collision repair loop).
 *
 * Use cases:
 * - Live hackathon demo: judge scans QR code, taps an object on the table,
 *   films 6 seconds of their hand moving it around an obstacle while narrating,
 *   and watches the SO-ARM101 reproduce and harden the task on the big screen.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  CheckCircle2,
  Compass,
  Crosshair,
  Mic,
  MicOff,
  Square,
  Upload,
  Video,
} from 'lucide-react';
import { APP_CONFIG } from '../config/appConfig';
import {
  fetchCompilerSession,
  uploadSessionVideo,
} from '../services/compilerApi';
import type { CompilerSessionState, MobileSensorCapture } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import {
  computeGravityFromEuler,
  requestMobileOrientationPermission,
  startAudioLevelMeter,
  subscribeMobileGravitySensor,
  type LiveOrientationReading,
} from '../utils/mobileSensorCapture';

interface MobileCaptureViewProps {
  sessionId: string;
  standalone: boolean;
  onUploadedToDesktop?: (session: CompilerSessionState) => void;
}

const DEFAULT_DEMO_SEC = APP_CONFIG.mobileHandDemo.defaultClipDurationSec;

/**
 * Renders the mobile-first QR capture viewfinder with tap-to-select reticle,
 * live DeviceOrientation gravity vector telemetry, microphone audio capture,
 * and a 6-second auto-stop hand-demonstration recorder.
 */
export const MobileCaptureView: React.FC<MobileCaptureViewProps> = ({
  sessionId,
  standalone,
  onUploadedToDesktop,
}) => {
  const nativeCameraRef = useRef<HTMLInputElement | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const gravitySamplesRef = useRef<Array<{ x: number; y: number; z: number }>>([]);
  const cleanupAudioMeterRef = useRef<(() => void) | null>(null);
  const cleanupGravityRef = useRef<(() => void) | null>(null);
  const localPreviewRef = useRef<string | null>(null);
  const recordStartMsRef = useRef<number>(0);

  const [session, setSession] = useState<CompilerSessionState | null>(null);
  const [captureMode, setCaptureMode] = useState<MobileSensorCapture['captureMode']>('hand_demo');
  const [taskGoal, setTaskGoal] = useState(
    'Move the tapped object cleanly around the tabletop obstacle and place it on the open side.',
  );
  const [autoStart, setAutoStart] = useState(true);
  const [autoStop6s, setAutoStop6s] = useState(true);

  const [isViewfinderOpen, setIsViewfinderOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [tapTarget, setTapTarget] = useState<{ x: number; y: number } | null>(null);
  const [orientation, setOrientation] = useState<LiveOrientationReading>(() =>
    computeGravityFromEuler(0, 48, 0),
  );
  const [hasLiveImu, setHasLiveImu] = useState(false);
  const [hasAudioTrack, setHasAudioTrack] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadedOk, setUploadedOk] = useState(false);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Hydrate paired desktop session
  useEffect(() => {
    logFunctionCall('MobileCaptureView.hydrateSession', { sessionId });
    fetchCompilerSession(sessionId)
      .then((s) => {
        setSession(s);
        if (s.taskGoal && s.taskGoal !== APP_CONFIG.simulation.defaultTaskGoal) {
          setTaskGoal(s.taskGoal);
        }
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Unable to pair with desktop session.');
      });
  }, [sessionId]);

  useEffect(() => {
    localPreviewRef.current = localPreviewUrl;
  }, [localPreviewUrl]);

  // Subscribe to DeviceOrientation / DeviceMotion immediately on browsers that allow passive listeners
  useEffect(() => {
    const unsub = subscribeMobileGravitySensor((reading) => {
      setOrientation(reading);
      setHasLiveImu(true);
      if (recorderRef.current && recorderRef.current.state === 'recording') {
        gravitySamplesRef.current.push(reading.gravityVector);
      }
    });
    cleanupGravityRef.current = unsub;
    return () => {
      unsub();
      cleanupGravityRef.current = null;
    };
  }, []);

  // Clean up camera, mic analyser, and blob URLs on unmount
  useEffect(() => {
    return () => {
      cleanupAudioMeterRef.current?.();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (localPreviewRef.current) {
        URL.revokeObjectURL(localPreviewRef.current);
        localPreviewRef.current = null;
      }
    };
  }, []);

  /**
   * Builds the finalized `MobileSensorCapture` payload (averaging gravity samples
   * collected across the 6-second clip for rock-solid horizon calibration).
   */
  const buildSensorSnapshot = useCallback(
    (recordedDurationSec: number, audioActive: boolean): MobileSensorCapture => {
      const samples = gravitySamplesRef.current;
      let avgGravity = orientation.gravityVector;
      if (samples.length > 0) {
        const sum = samples.reduce(
          (acc, s) => ({ x: acc.x + s.x, y: acc.y + s.y, z: acc.z + s.z }),
          { x: 0, y: 0, z: 0 },
        );
        avgGravity = {
          x: Number((sum.x / samples.length).toFixed(3)),
          y: Number((sum.y / samples.length).toFixed(3)),
          z: Number((sum.z / samples.length).toFixed(3)),
        };
      }

      return {
        tapTarget,
        gravityVector: avgGravity,
        orientationDeg: orientation.orientationDeg,
        hasAudioTrack: audioActive,
        clipDurationSec: Number(Math.max(1, recordedDurationSec).toFixed(1)),
        captureMode,
        capturedAt: new Date().toISOString(),
      };
    },
    [captureMode, orientation, tapTarget],
  );

  /**
   * Streams the recorded clip + DeviceOrientation gravity vector + tap target +
   * audio metadata to the paired desktop session.
   */
  const sendClipToSession = useCallback(
    async (
      file: File | Blob,
      filename: string,
      recordedDurationSec: number,
      audioActive: boolean,
    ): Promise<void> => {
      const sensors = buildSensorSnapshot(recordedDurationSec, audioActive);
      logFunctionCall('MobileCaptureView.sendClipToSession', {
        sessionId,
        filename,
        sizeBytes: file.size,
        autoStart,
        captureMode: sensors.captureMode,
        hasTapTarget: Boolean(sensors.tapTarget),
        hasAudioTrack: sensors.hasAudioTrack,
        gravityVector: sensors.gravityVector,
      });

      setError(null);
      setUploadedOk(false);
      setUploadProgress(0);
      setLocalPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });

      try {
        const result = await uploadSessionVideo({
          sessionId,
          file,
          filename,
          source: 'mobile_qr',
          taskGoal,
          autoStart,
          mobileSensors: sensors,
          onProgress: (pct) => setUploadProgress(pct),
        });
        setSession(result.session);
        setUploadedOk(true);
        if (onUploadedToDesktop) {
          onUploadedToDesktop(result.session);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed.');
      } finally {
        setUploadProgress(null);
      }
    },
    [autoStart, buildSensorSnapshot, onUploadedToDesktop, sessionId, taskGoal],
  );

  /**
   * Stops active MediaRecorder capture (triggers automatic upload in `onstop`).
   */
  const handleStopRecording = useCallback((): void => {
    logFunctionCall('MobileCaptureView.handleStopRecording', { elapsedSec });
    if (recorderRef.current && recorderRef.current.state === 'recording') {
      recorderRef.current.stop();
    }
  }, [elapsedSec]);

  // 100ms precision timer + automatic 6.0-second stop for judge hand demos
  useEffect(() => {
    if (!isRecording) {
      setElapsedSec(0);
      return;
    }
    const interval = window.setInterval(() => {
      const sec = (Date.now() - recordStartMsRef.current) / 1000;
      setElapsedSec(sec);
      if (autoStop6s && captureMode === 'hand_demo' && sec >= DEFAULT_DEMO_SEC) {
        handleStopRecording();
      }
    }, 100);
    return () => window.clearInterval(interval);
  }, [isRecording, autoStop6s, captureMode, handleStopRecording]);

  /**
   * Opens the live rear camera viewfinder WITH microphone audio track and
   * requests iOS/Android DeviceOrientation IMU permission so the judge can tap
   * the object on the table before filming the 6-second clip.
   */
  const handleOpenViewfinder = async (): Promise<void> => {
    logFunctionCall('MobileCaptureView.handleOpenViewfinder');
    setError(null);

    // Request iOS 13+ DeviceOrientation / DeviceMotion permission on user tap
    const imuPermitted = await requestMobileOrientationPermission();
    if (imuPermitted) {
      cleanupGravityRef.current?.();
      cleanupGravityRef.current = subscribeMobileGravitySensor((reading) => {
        setOrientation(reading);
        setHasLiveImu(true);
        if (recorderRef.current && recorderRef.current.state === 'recording') {
          gravitySamplesRef.current.push(reading.gravityVector);
        }
      });
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Live camera viewfinder is unavailable here; use Native Rear Camera below.');
      return;
    }

    try {
      let stream: MediaStream;
      let micActive = false;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: APP_CONFIG.mobileHandDemo.enableAudioCapture
            ? { echoCancellation: true, noiseSuppression: true }
            : false,
        });
        micActive = stream.getAudioTracks().length > 0;
      } catch {
        // Gracefully fall back to video-only if microphone permission was denied
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        micActive = false;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      streamRef.current = stream;
      setHasAudioTrack(micActive);
      setIsViewfinderOpen(true);

      cleanupAudioMeterRef.current?.();
      cleanupAudioMeterRef.current = startAudioLevelMeter(stream, setAudioLevel);

      window.setTimeout(async () => {
        if (videoPreviewRef.current) {
          videoPreviewRef.current.srcObject = stream;
          try {
            await videoPreviewRef.current.play();
          } catch {
            // Autoplay handled by playsInline + muted
          }
        }
      }, 40);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open rear camera.');
    }
  };

  /**
   * Starts recording the 6-second hand demonstration (video + audio track + IMU).
   */
  const handleStartRecording = async (): Promise<void> => {
    logFunctionCall('MobileCaptureView.handleStartRecording', {
      captureMode,
      tapTarget,
      autoStop6s,
    });
    setError(null);
    if (!streamRef.current) {
      await handleOpenViewfinder();
    }
    const stream = streamRef.current;
    if (!stream) return;

    chunksRef.current = [];
    gravitySamplesRef.current = [orientation.gravityVector];
    recordStartMsRef.current = Date.now();

    const preferredMime = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
      ? 'video/webm;codecs=vp8,opus'
      : MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
        ? 'video/webm;codecs=vp8'
        : MediaRecorder.isTypeSupported('video/mp4')
          ? 'video/mp4'
          : '';

    const recorder = new MediaRecorder(
      stream,
      preferredMime ? { mimeType: preferredMime } : undefined,
    );
    recorderRef.current = recorder;

    const micWasActive = stream.getAudioTracks().length > 0;
    recorder.ondataavailable = (ev) => {
      if (ev.data.size > 0) chunksRef.current.push(ev.data);
    };
    recorder.onstop = async () => {
      const actualSec = Math.max(1, (Date.now() - recordStartMsRef.current) / 1000);
      const mime = recorder.mimeType || 'video/webm';
      const ext = mime.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      setIsRecording(false);
      if (blob.size > 0) {
        await sendClipToSession(
          blob,
          `hand_demo_${Date.now()}.${ext}`,
          actualSec,
          micWasActive,
        );
      }
    };

    recorder.start(200);
    setIsRecording(true);
  };

  /**
   * Records the normalized `[0..1, 0..1]` tap coordinate directly on the live
   * viewfinder so Gemini 4 Carbon locks onto the tapped tabletop object.
   */
  const handleTapViewfinder = (e: React.MouseEvent<HTMLDivElement>): void => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const next = { x: Number(x.toFixed(4)), y: Number(y.toFixed(4)) };
    logFunctionCall('MobileCaptureView.tapTarget', next);
    setTapTarget(next);
  };

  const progressPct =
    autoStop6s && captureMode === 'hand_demo'
      ? Math.min(100, (elapsedSec / DEFAULT_DEMO_SEC) * 100)
      : 100;

  return (
    <div
      className={
        standalone
          ? 'min-h-screen bg-[#07090E] text-slate-100 flex flex-col'
          : 'max-w-lg mx-auto bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden'
      }
    >
      {/* Compact Header (<= 15% mobile sticky cap) */}
      <div className="px-4 py-3 bg-[#0B0E17] border-b border-[#1E293B] flex items-center justify-between gap-3">
        <div>
          <h1 className="font-display text-base font-bold text-slate-100">
            LetThereBeSims Hand Demo Capture
          </h1>
          <p className="font-mono-tabular text-[11px] text-cyan-400">
            Paired · {sessionId} · SO-ARM101
          </p>
        </div>
        <span className="font-mono-tabular text-[11px] text-emerald-400">
          {session ? '✓ Linked' : 'Connecting'}
        </span>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1">
        {/* Capture Mode Switcher */}
        <div className="grid grid-cols-2 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
          <button
            type="button"
            onClick={() => setCaptureMode('hand_demo')}
            className={`min-h-[40px] px-3 text-xs font-medium rounded transition-colors cursor-pointer ${
              captureMode === 'hand_demo'
                ? 'bg-amber-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            6s Hand Demo → SO-ARM101
          </button>
          <button
            type="button"
            onClick={() => setCaptureMode('robot_footage')}
            className={`min-h-[40px] px-3 text-xs font-medium rounded transition-colors cursor-pointer ${
              captureMode === 'robot_footage'
                ? 'bg-amber-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Robot Failure / Demo
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-md bg-rose-500/10 border border-rose-500/35 text-xs text-rose-200">
            {error}
          </div>
        )}

        {uploadedOk && (
          <div className="p-3.5 rounded-md bg-emerald-500/12 border border-emerald-500/35 flex items-start gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div className="text-xs text-emerald-100 leading-relaxed">
              <p className="font-semibold">
                Video + Gravity Vector + Audio Streamed to Desktop
              </p>
              <p className="text-emerald-200/80 mt-0.5">
                {autoStart
                  ? 'Gemini 4 Carbon is building the MuJoCo scene & retargeting your hand trajectory to SO-ARM101, followed by Hydrogen 20-env rollouts & Neon repair.'
                  : 'Staged on desktop. Tap Analyze & Verify Policy to compile.'}
              </p>
            </div>
          </div>
        )}

        {/* Interactive Tap-to-Select Live Viewfinder */}
        <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg overflow-hidden flex flex-col">
          <div className="px-3.5 py-2.5 border-b border-[#1E293B] flex items-center justify-between gap-2 text-xs">
            <span className="font-medium text-slate-200">
              {isViewfinderOpen
                ? '1. Tap Object on Table · 2. Film 6s Arc'
                : 'Live Camera + Gravity + Audio Viewfinder'}
            </span>
            {isRecording ? (
              <span className="font-mono-tabular text-rose-400 font-semibold">
                REC {elapsedSec.toFixed(1)}s{autoStop6s && captureMode === 'hand_demo' ? ` / ${DEFAULT_DEMO_SEC}.0s` : ''}
              </span>
            ) : (
              <span className="font-mono-tabular text-[11px] text-slate-400">
                {tapTarget
                  ? `Target (${Math.round(tapTarget.x * 100)}%, ${Math.round(tapTarget.y * 100)}%)`
                  : 'Tap to lock object'}
              </span>
            )}
          </div>

          <div
            role="button"
            tabIndex={0}
            aria-label="Tap the target object on the tabletop"
            onClick={handleTapViewfinder}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                setTapTarget({ x: 0.5, y: 0.55 });
              }
            }}
            className="relative w-full aspect-[4/3] bg-black overflow-hidden select-none cursor-crosshair"
          >
            <video
              ref={videoPreviewRef}
              muted
              playsInline
              className={`w-full h-full object-cover ${isViewfinderOpen ? 'block' : 'hidden'}`}
            />

            {!isViewfinderOpen && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center gap-2">
                <Crosshair className="w-8 h-8 text-amber-400" />
                <p className="text-sm font-medium text-slate-200">
                  Open Viewfinder to Tap Object &amp; Film 6s Demo
                </p>
                <p className="text-xs text-slate-400 max-w-xs">
                  Captures video + spoken audio track + DeviceOrientation gravity vector so Carbon calibrates the tabletop and retargets your hand path to SO-ARM101.
                </p>
              </div>
            )}

            {/* Tap-to-Select Target Reticle Overlay */}
            {tapTarget && (
              <div
                style={{
                  left: `${tapTarget.x * 100}%`,
                  top: `${tapTarget.y * 100}%`,
                }}
                className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center"
              >
                <div className="w-12 h-12 rounded-full border-2 border-amber-400 bg-amber-400/10 flex items-center justify-center">
                  <div className="w-2 h-2 rounded-full bg-amber-400" />
                </div>
                <span className="mt-1 px-1.5 py-0.5 rounded bg-black/80 font-mono-tabular text-[10px] text-amber-300 whitespace-nowrap">
                  TARGET ({Math.round(tapTarget.x * 100)}%, {Math.round(tapTarget.y * 100)}%)
                </span>
              </div>
            )}

            {/* 6-Second Countdown Progress Bar */}
            {isRecording && (
              <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-black/70">
                <div
                  className="h-full bg-rose-500 transition-[width] duration-100"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            )}
          </div>

          {/* Live Gravity Vector & Microphone Audio Telemetry Strip */}
          <div className="px-3.5 py-2.5 border-t border-[#1E293B] bg-[#0B0E17] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono-tabular text-slate-300">
            <span className="inline-flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span>
                g=({orientation.gravityVector.x.toFixed(1)}, {orientation.gravityVector.y.toFixed(1)},{' '}
                {orientation.gravityVector.z.toFixed(1)})
              </span>
              <span className="text-slate-500">·</span>
              <span className="text-slate-400">
                tilt {orientation.orientationDeg.cameraTiltDeg.toFixed(0)}°
                {hasLiveImu ? ' (IMU)' : ' (est)'}
              </span>
            </span>

            <span className="inline-flex items-center gap-1.5">
              {hasAudioTrack ? (
                <>
                  <Mic className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span className="text-emerald-300">Mic</span>
                  <span className="w-12 h-1.5 bg-[#161D2C] rounded overflow-hidden inline-block">
                    <span
                      className="block h-full bg-emerald-400"
                      style={{ width: `${Math.max(10, Math.round(audioLevel * 100))}%` }}
                    />
                  </span>
                </>
              ) : (
                <>
                  <MicOff className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  <span className="text-slate-400">Mic pending</span>
                </>
              )}
            </span>
          </div>
        </div>

        {/* Thumb-Zone Primary Actions (>= 48px height) */}
        <div className="flex flex-col gap-2.5">
          {!isViewfinderOpen ? (
            <button
              type="button"
              disabled={uploadProgress !== null}
              onClick={() => void handleOpenViewfinder()}
              className="w-full min-h-[54px] px-4 py-3.5 inline-flex items-center justify-center gap-2.5 text-sm font-semibold text-slate-950 bg-amber-500 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors cursor-pointer"
            >
              <Video className="w-5 h-5" />
              <span>Enable Camera, Mic &amp; Gravity Sensor</span>
            </button>
          ) : isRecording ? (
            <button
              type="button"
              onClick={handleStopRecording}
              className="w-full min-h-[54px] px-4 py-3.5 inline-flex items-center justify-center gap-2.5 text-sm font-semibold text-white bg-rose-600 rounded-lg hover:bg-rose-500 transition-colors cursor-pointer"
            >
              <Square className="w-4 h-4" />
              <span>Stop &amp; Retarget Now ({elapsedSec.toFixed(1)}s)</span>
            </button>
          ) : (
            <button
              type="button"
              disabled={uploadProgress !== null}
              onClick={() => void handleStartRecording()}
              className="w-full min-h-[54px] px-4 py-3.5 inline-flex items-center justify-center gap-2.5 text-sm font-semibold text-slate-950 bg-amber-500 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors cursor-pointer"
            >
              <Camera className="w-5 h-5" />
              <span>
                {captureMode === 'hand_demo' && autoStop6s
                  ? `Film ${DEFAULT_DEMO_SEC}s Hand Demo (Audio + Gravity)`
                  : 'Start Recording (Audio + Gravity)'}
              </span>
            </button>
          )}

          {/* Native Rear Camera Fallback */}
          <input
            ref={nativeCameraRef}
            type="file"
            accept="video/*,.mp4,.webm,.mov"
            capture="environment"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = '';
              if (file) {
                void sendClipToSession(file, file.name, DEFAULT_DEMO_SEC, true);
              }
            }}
            className="hidden"
          />
          <button
            type="button"
            disabled={uploadProgress !== null || isRecording}
            onClick={() => nativeCameraRef.current?.click()}
            className="w-full min-h-[46px] px-4 py-2.5 inline-flex items-center justify-center gap-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded-lg hover:border-[#2D3B52] disabled:opacity-50 transition-colors cursor-pointer"
          >
            <Upload className="w-4 h-4" />
            <span>Or Upload Native Rear-Camera Clip</span>
          </button>
        </div>

        {/* Upload Progress Bar */}
        {uploadProgress !== null && (
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg p-3.5 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-mono-tabular text-amber-300">
              <span className="inline-flex items-center gap-1.5">
                <Upload className="w-3.5 h-3.5" />
                Streaming video + gravity + audio...
              </span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full h-2 bg-[#161D2C] rounded overflow-hidden">
              <div
                className="h-full bg-amber-500"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Captured Clip Preview */}
        {localPreviewUrl && (
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg p-2.5 flex flex-col gap-1.5">
            <span className="text-[11px] font-medium text-slate-300">Last Captured Clip</span>
            <video
              key={localPreviewUrl}
              src={localPreviewUrl}
              controls
              playsInline
              preload="auto"
              className="w-full h-36 object-contain rounded bg-black border border-[#1E293B]"
            />
          </div>
        )}

        {/* Task Goal & Options */}
        <div className="flex flex-col gap-2.5 pt-1">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="mobile-goal" className="text-xs font-medium text-slate-300">
              Task Goal (or speak it while filming)
            </label>
            <textarea
              id="mobile-goal"
              rows={2}
              value={taskGoal}
              onChange={(e) => setTaskGoal(e.target.value)}
              className="w-full px-3 py-2 text-sm text-slate-100 bg-[#0B0E17] border border-[#1E293B] rounded-md focus:outline-none focus:border-amber-500 resize-none"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <label className="flex items-center justify-between gap-2 px-3 py-2.5 bg-[#0B0E17] border border-[#1E293B] rounded-md cursor-pointer min-h-[44px]">
              <span className="text-xs text-slate-200">Auto-stop at 6s</span>
              <input
                type="checkbox"
                checked={autoStop6s}
                onChange={(e) => setAutoStop6s(e.target.checked)}
                className="w-4 h-4 accent-amber-500"
              />
            </label>

            <label className="flex items-center justify-between gap-2 px-3 py-2.5 bg-[#0B0E17] border border-[#1E293B] rounded-md cursor-pointer min-h-[44px]">
              <span className="text-xs text-slate-200">Auto-run Carbon → Neon</span>
              <input
                type="checkbox"
                checked={autoStart}
                onChange={(e) => setAutoStart(e.target.checked)}
                className="w-4 h-4 accent-amber-500"
              />
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};
