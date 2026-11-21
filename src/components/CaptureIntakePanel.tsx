/**
 * Video Capture & QR Smartphone Pairing Panel (`CaptureIntakePanel`)
 *
 * Lives in the top half of the left command rail. Provides three seamless ways
 * to feed real-world tabletop footage (failure clips, successful demonstrations,
 * or robot-free workspace scans) into the Gemini 4 pipeline:
 * 1. Live scannable QR code pointing to `${window.location.origin}/capture?session=<id>`
 *    so any robotics engineer can record on their phone and watch the desktop
 *    stage react automatically via SSE.
 * 2. Desktop drag-and-drop / multi-file picker supporting either **1 video**
 *    (Single View) or **2 videos at the same time** (View 1 Primary + optional
 *    View 2 Secondary/Wrist angle, unlocking the Dual 3D View automatically).
 * 3. Built-in desktop webcam `MediaRecorder` for instant 5–10 second captures.
 *
 * Also displays the detected execution outcome (`Success`, `Failure`, or
 * `No Robot / Scene Map`) once Gemini 4 Carbon finishes Stage 1, alongside the
 * robot platform selector (`SO-ARM101` vs `Franka Panda`) and task goal input.
 *
 * Use cases:
 * - Uploading a single robot video (failure or success) for Single View analysis.
 * - Selecting or dropping 2 synchronized camera angles (e.g., Overhead + Wrist
 *   Cam) simultaneously to unlock Dual View 3D inspection & multi-view fusion.
 * - Optionally adding or removing a 2nd camera view at any time.
 */

import React, { useEffect, useRef, useState } from 'react';
import QRCode from 'qrcode';
import {
  Camera,
  Check,
  Copy,
  ExternalLink,
  Play,
  Plus,
  Square,
  Upload,
  X,
} from 'lucide-react';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type { CompilerSessionState, VideoSlot } from '../types/robotics';
import { copyTextToClipboard } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';
import { usePipelineProgress } from '../utils/pipelineProgress';

export interface LocalVideoMeta {
  name: string;
  mimeType: string;
  sizeBytes: number;
}

export type RolloutDurationSelection = 'match' | 15 | 30 | 60;

interface CaptureIntakePanelProps {
  session: CompilerSessionState;
  taskGoal: string;
  robotModel: RobotModelId;
  durationMode: RolloutDurationSelection;
  detectedVideoDurationSec: number | null;
  uploadProgress: number | null;
  uploadStatusLabel?: string | null;
  isUploading: boolean;
  localVideoPreviewUrl?: string | null;
  localVideoMeta?: LocalVideoMeta | null;
  localSecondaryVideoPreviewUrl?: string | null;
  localSecondaryVideoMeta?: LocalVideoMeta | null;
  uploadError?: string | null;
  onTaskGoalChange: (val: string) => void;
  onRobotModelChange: (model: RobotModelId) => void;
  onDurationModeChange: (mode: RolloutDurationSelection) => void;
  onDetectedVideoDuration: (durationSec: number | null) => void;
  onUploadFiles: (
    files: Array<{ file: File | Blob; filename: string; slot?: VideoSlot }>,
    source: 'desktop_upload' | 'desktop_webcam',
  ) => Promise<void>;
  onRemoveSecondaryVideo: () => Promise<void>;
  onRunPipeline: () => Promise<void>;
  onCancelPipeline: () => Promise<void>;
  onOpenMobilePreview: () => void;
}

/**
 * Reusable preview tile for View 1 (Primary) or View 2 (Optional Secondary)
 * with instant local blob playback, automatic server fallback, duration badge,
 * and codec-unavailable notice.
 */
const MediaPreviewTile: React.FC<{
  label: string;
  localUrl: string | null;
  serverUrl: string | null;
  name: string;
  mimeType: string;
  sizeBytes: number;
  isUploading: boolean;
  compact: boolean;
  onDurationDetected?: (durationSec: number) => void;
  onRemove?: () => void;
}> = ({
  label,
  localUrl,
  serverUrl,
  name,
  mimeType,
  sizeBytes,
  isUploading,
  compact,
  onDurationDetected,
  onRemove,
}) => {
  const [useServerFallback, setUseServerFallback] = useState(false);
  const [playbackError, setPlaybackError] = useState(false);
  const [durationSec, setDurationSec] = useState<number | null>(null);

  const previewUrl = useServerFallback ? serverUrl : localUrl || serverUrl;

  useEffect(() => {
    setUseServerFallback(false);
    setPlaybackError(false);
    setDurationSec(null);
  }, [localUrl, serverUrl]);

  if (!previewUrl) return null;

  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex items-center justify-between gap-1.5 text-[11px] text-slate-300">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="font-mono-tabular text-[10px] text-amber-400 uppercase shrink-0">
            {label}
          </span>
          <span className="truncate font-medium" title={name}>
            {name}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="font-mono-tabular text-[10px] text-emerald-400">
            {isUploading
              ? 'Uploading...'
              : `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB${durationSec !== null ? ` · ${durationSec.toFixed(1)}s` : ''}`}
          </span>
          {onRemove && (
            <button
              type="button"
              onClick={onRemove}
              title="Remove View 2 and return to Single View"
              aria-label="Remove secondary video"
              className="w-4 h-4 inline-flex items-center justify-center rounded bg-[#161D2C] border border-[#2D3B52] text-slate-400 hover:text-rose-300 hover:border-rose-500/50 transition-colors cursor-pointer"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          )}
        </div>
      </div>

      {mimeType.startsWith('image/') ? (
        <img
          src={previewUrl}
          alt={`${label} uploaded tabletop frame`}
          referrerPolicy="no-referrer"
          className={`w-full ${compact ? 'h-24' : 'h-36'} object-cover rounded border border-[#1E293B] bg-black`}
        />
      ) : playbackError ? (
        <div className="w-full p-2.5 rounded border border-amber-500/30 bg-[#111622] text-[10px] text-slate-300 leading-relaxed">
          <p className="font-medium text-amber-300">Codec preview unavailable</p>
          <p className="mt-0.5 text-slate-400">Uploaded and ready for Gemini 4.</p>
        </div>
      ) : (
        <video
          key={previewUrl}
          src={previewUrl}
          controls
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={(e) => {
            const dur = e.currentTarget.duration;
            if (Number.isFinite(dur) && dur > 0) {
              setDurationSec(dur);
              onDurationDetected?.(dur);
            }
          }}
          onError={() => {
            if (!useServerFallback && localUrl && serverUrl) {
              setUseServerFallback(true);
            } else {
              setPlaybackError(true);
            }
          }}
          className={`w-full ${compact ? 'h-24' : 'h-36'} object-contain rounded bg-black border border-[#1E293B]`}
        />
      )}
    </div>
  );
};

/**
 * Renders the left-rail capture intake, live QR code pairer, 1-or-2 video
 * uploader, webcam recorder, detected execution outcome summary, task goal
 * prompt, and Compile Verified Policy trigger.
 */
export const CaptureIntakePanel: React.FC<CaptureIntakePanelProps> = ({
  session,
  taskGoal,
  robotModel,
  durationMode,
  detectedVideoDurationSec,
  uploadProgress,
  uploadStatusLabel = null,
  isUploading,
  localVideoPreviewUrl = null,
  localVideoMeta = null,
  localSecondaryVideoPreviewUrl = null,
  localSecondaryVideoMeta = null,
  uploadError = null,
  onTaskGoalChange,
  onRobotModelChange,
  onDurationModeChange,
  onDetectedVideoDuration,
  onUploadFiles,
  onRemoveSecondaryVideo,
  onRunPipeline,
  onCancelPipeline,
  onOpenMobilePreview,
}) => {
  const qrCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const secondaryFileInputRef = useRef<HTMLInputElement | null>(null);
  const webcamVideoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const [copiedLink, setCopiedLink] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isRecordingWebcam, setIsRecordingWebcam] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const hasPrimary = Boolean(localVideoPreviewUrl || session.video);
  const hasSecondary = Boolean(localSecondaryVideoPreviewUrl || session.secondaryVideo);

  const primaryName = session.video?.originalName ?? localVideoMeta?.name ?? '';
  const primaryMime = session.video?.mimeType ?? localVideoMeta?.mimeType ?? 'video/mp4';
  const primarySize = session.video?.sizeBytes ?? localVideoMeta?.sizeBytes ?? 0;

  const secondaryName = session.secondaryVideo?.originalName ?? localSecondaryVideoMeta?.name ?? '';
  const secondaryMime =
    session.secondaryVideo?.mimeType ?? localSecondaryVideoMeta?.mimeType ?? 'video/mp4';
  const secondarySize =
    session.secondaryVideo?.sizeBytes ?? localSecondaryVideoMeta?.sizeBytes ?? 0;

  const mobileCaptureUrl =
    typeof window !== 'undefined'
      ? `${window.location.origin}/capture?session=${encodeURIComponent(session.id)}`
      : `/capture?session=${encodeURIComponent(session.id)}`;

  // Render crisp scannable QR code onto the local canvas whenever session changes
  useEffect(() => {
    if (!qrCanvasRef.current) return;
    logFunctionCall('CaptureIntakePanel.renderQrCode', { sessionId: session.id });
    void QRCode.toCanvas(qrCanvasRef.current, mobileCaptureUrl, {
      width: 112,
      margin: 1,
      color: {
        dark: '#07090E',
        light: '#F8FAFC',
      },
    });
  }, [mobileCaptureUrl, session.id]);

  // Clean up active webcam tracks on unmount
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Increment webcam timer while recording
  useEffect(() => {
    if (!isRecordingWebcam) {
      setRecordingSeconds(0);
      return;
    }
    const timer = window.setInterval(() => {
      setRecordingSeconds((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isRecordingWebcam]);

  /**
   * Copies the live mobile capture URL to clipboard for rapid phone pairing.
   */
  const handleCopyMobileUrl = async (): Promise<void> => {
    logFunctionCall('CaptureIntakePanel.handleCopyMobileUrl', { mobileCaptureUrl });
    const ok = await copyTextToClipboard(mobileCaptureUrl);
    if (ok) {
      setCopiedLink(true);
      window.setTimeout(() => setCopiedLink(false), 1800);
    }
  };

  /**
   * Handles desktop drag-and-drop or multi-file picker selection (1 file sets
   * View 1 / Single View; 2 files selected together set View 1 + View 2 / Dual View).
   */
  const handleFilesSelected = async (
    fileList: FileList | null,
    forcedSlot?: VideoSlot,
  ): Promise<void> => {
    if (!fileList || fileList.length === 0) return;
    const picked = Array.from(fileList).slice(0, forcedSlot ? 1 : 2);
    logFunctionCall('CaptureIntakePanel.handleFilesSelected', {
      count: picked.length,
      forcedSlot: forcedSlot ?? null,
      names: picked.map((f) => f.name),
    });
    setWebcamError(null);

    if (forcedSlot === 'secondary') {
      await onUploadFiles(
        [{ file: picked[0], filename: picked[0].name, slot: 'secondary' }],
        'desktop_upload',
      );
      return;
    }

    const uploads: Array<{ file: File | Blob; filename: string; slot: VideoSlot }> = [
      { file: picked[0], filename: picked[0].name, slot: 'primary' },
    ];
    if (picked.length >= 2) {
      uploads.push({ file: picked[1], filename: picked[1].name, slot: 'secondary' });
    }
    await onUploadFiles(uploads, 'desktop_upload');
  };

  /**
   * Starts a live desktop webcam MediaRecorder capture session.
   */
  const handleStartWebcam = async (): Promise<void> => {
    logFunctionCall('CaptureIntakePanel.handleStartWebcam');
    setWebcamError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setWebcamError('Camera capture is unavailable in this browser context.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 960 }, height: { ideal: 540 } },
        audio: false,
      });
      mediaStreamRef.current = stream;
      if (webcamVideoRef.current) {
        webcamVideoRef.current.srcObject = stream;
        await webcamVideoRef.current.play();
      }
      recordedChunksRef.current = [];
      const preferredMime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType: preferredMime });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (evt) => {
        if (evt.data.size > 0) recordedChunksRef.current.push(evt.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        stream.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
        setIsRecordingWebcam(false);
        if (blob.size > 0) {
          await onUploadFiles(
            [{ file: blob, filename: `webcam_capture_${Date.now()}.webm`, slot: 'primary' }],
            'desktop_webcam',
          );
        }
      };

      recorder.start(250);
      setIsRecordingWebcam(true);
    } catch (err) {
      setWebcamError(
        err instanceof Error ? err.message : 'Unable to access webcam. Check browser permissions.',
      );
    }
  };

  /**
   * Stops the active desktop webcam MediaRecorder and uploads the resulting clip.
   */
  const handleStopWebcam = (): void => {
    logFunctionCall('CaptureIntakePanel.handleStopWebcam', { recordingSeconds });
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const canCompile =
    Boolean(session.video) && !isUploading && session.status !== 'running';

  const detectedOutcome = session.sceneSpec?.executionOutcome ?? null;
  const pipelineProgress = usePipelineProgress(session);

  return (
    <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-4">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="font-display text-sm font-semibold text-slate-100">
          01. Capture & Session Intake
        </h2>
        <span className="font-mono-tabular text-[11px] text-slate-400">
          {session.id}
        </span>
      </div>

      {/* QR Smartphone Pairer */}
      <div className="flex items-center gap-3.5 pb-4 border-b border-[#1E293B]">
        <div className="bg-slate-50 p-1.5 rounded-md shrink-0">
          <canvas
            ref={qrCanvasRef}
            width={112}
            height={112}
            aria-label="QR code for smartphone video capture"
            className="block w-[96px] h-[96px]"
          />
        </div>
        <div className="flex flex-col gap-1.5 min-w-0">
          <p className="text-xs font-medium text-slate-200">
            Scan with Smartphone
          </p>
          <p className="text-[11px] leading-relaxed text-slate-400">
            Record any robot run (success or failure) or robot-free tabletop. Footage streams directly to this command center.
          </p>
          <div className="flex items-center gap-2 pt-0.5">
            <button
              type="button"
              onClick={handleCopyMobileUrl}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded hover:border-[#2D3B52] transition-colors whitespace-nowrap cursor-pointer"
            >
              {copiedLink ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              <span>{copiedLink ? 'Copied' : 'Copy Link'}</span>
            </button>
            <button
              type="button"
              onClick={onOpenMobilePreview}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-cyan-300 bg-[#161D2C] border border-[#1E293B] rounded hover:border-cyan-500/40 transition-colors whitespace-nowrap cursor-pointer"
            >
              <ExternalLink className="w-3 h-3" />
              <span>Mobile UI</span>
            </button>
          </div>
        </div>
      </div>

      {/* Desktop Drag & Drop Upload (1 or 2 Videos) + Webcam Recorder */}
      <div className="flex flex-col gap-2">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            void handleFilesSelected(e.dataTransfer.files);
          }}
          className={`border border-dashed rounded-md p-3 transition-colors ${
            isDragging
              ? 'border-amber-400 bg-amber-500/10'
              : 'border-[#2D3B52] bg-[#0B0E17]'
          }`}
        >
          {/* Primary input accepts 1 OR 2 files simultaneously */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="video/*,.mp4,.webm,.mov,.m4v,image/jpeg,image/png"
            onChange={(e) => {
              const files = e.target.files;
              void handleFilesSelected(files).finally(() => {
                if (fileInputRef.current) fileInputRef.current.value = '';
              });
            }}
            className="hidden"
          />

          {/* Optional Secondary View input */}
          <input
            ref={secondaryFileInputRef}
            type="file"
            accept="video/*,.mp4,.webm,.mov,.m4v,image/jpeg,image/png"
            onChange={(e) => {
              const files = e.target.files;
              void handleFilesSelected(files, 'secondary').finally(() => {
                if (secondaryFileInputRef.current) secondaryFileInputRef.current.value = '';
              });
            }}
            className="hidden"
          />

          {isRecordingWebcam ? (
            <div className="flex flex-col gap-2">
              <video
                ref={webcamVideoRef}
                muted
                playsInline
                className="w-full h-32 object-cover rounded bg-black border border-[#1E293B]"
              />
              <div className="flex items-center justify-between">
                <span className="font-mono-tabular text-xs text-rose-400 font-medium">
                  REC 00:{String(recordingSeconds).padStart(2, '0')}
                </span>
                <button
                  type="button"
                  onClick={handleStopWebcam}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-rose-600 rounded hover:bg-rose-500 transition-colors cursor-pointer"
                >
                  <Square className="w-3 h-3" />
                  <span>Stop & Upload</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-2.5">
              <div className="flex items-center justify-between gap-2">
                <button
                  type="button"
                  disabled={isUploading}
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded hover:border-[#2D3B52] disabled:opacity-50 transition-colors whitespace-nowrap cursor-pointer"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload 1 or 2 Videos</span>
                </button>
                <button
                  type="button"
                  disabled={isUploading}
                  onClick={() => void handleStartWebcam()}
                  className="inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded hover:border-[#2D3B52] disabled:opacity-50 transition-colors whitespace-nowrap cursor-pointer"
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Webcam</span>
                </button>
              </div>

              {isUploading && uploadProgress !== null && (
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[11px] font-mono-tabular text-amber-300">
                    <span>{uploadStatusLabel || 'Uploading footage...'}</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-[#161D2C] rounded overflow-hidden">
                    <div
                      className="h-full bg-amber-500 transition-transform"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Staged Video Preview(s): Single Full-Width for 1 Video, Split Grid for 2 Videos */}
              {hasPrimary && (
                <div className="pt-1 flex flex-col gap-2">
                  <div
                    className={
                      hasSecondary
                        ? 'grid grid-cols-2 gap-2'
                        : 'flex flex-col gap-2'
                    }
                  >
                    <MediaPreviewTile
                      label="View 1"
                      localUrl={localVideoPreviewUrl}
                      serverUrl={session.video?.mediaUrl ?? null}
                      name={primaryName}
                      mimeType={primaryMime}
                      sizeBytes={primarySize}
                      isUploading={isUploading && !session.video}
                      compact={hasSecondary}
                      onDurationDetected={(dur) => onDetectedVideoDuration(dur)}
                    />
                    {hasSecondary && (
                      <MediaPreviewTile
                        label="View 2"
                        localUrl={localSecondaryVideoPreviewUrl}
                        serverUrl={session.secondaryVideo?.mediaUrl ?? null}
                        name={secondaryName}
                        mimeType={secondaryMime}
                        sizeBytes={secondarySize}
                        isUploading={isUploading && !session.secondaryVideo}
                        compact
                        onRemove={() => void onRemoveSecondaryVideo()}
                      />
                    )}
                  </div>

                  {!hasSecondary && !isUploading && (
                    <button
                      type="button"
                      onClick={() => secondaryFileInputRef.current?.click()}
                      className="w-full py-1.5 px-2.5 inline-flex items-center justify-center gap-1.5 text-[11px] font-medium text-slate-300 bg-[#111622] border border-dashed border-[#2D3B52] rounded hover:border-cyan-500/50 hover:text-cyan-200 transition-colors cursor-pointer"
                    >
                      <Plus className="w-3 h-3" />
                      <span>Add Optional View 2 (Unlocks Dual 3D View)</span>
                    </button>
                  )}
                </div>
              )}

              {!hasPrimary && !isUploading && (
                <p className="text-[11px] text-slate-400 text-center leading-relaxed">
                  Drop 1 video (Single View) or 2 camera angles at once (Dual View) · MP4, WebM, MOV (max 60 MB each)
                </p>
              )}
            </div>
          )}
        </div>

        {(webcamError || uploadError) && (
          <p className="text-xs text-rose-400 leading-relaxed">{webcamError || uploadError}</p>
        )}
      </div>

      {/* Detected Outcome Summary (Populated after Stage 1 Carbon Detection) */}
      {session.sceneSpec && detectedOutcome && (
        <div
          className={`rounded-md p-2.5 border text-xs leading-relaxed ${
            detectedOutcome === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/35 text-emerald-100'
              : detectedOutcome === 'no_robot'
                ? 'bg-cyan-500/10 border-cyan-500/35 text-cyan-100'
                : 'bg-rose-500/10 border-rose-500/35 text-rose-100'
          }`}
        >
          <div className="flex items-center justify-between gap-2 font-mono-tabular text-[11px] font-semibold">
            <span>
              {detectedOutcome === 'success'
                ? '✓ Successful Demo Detected'
                : detectedOutcome === 'no_robot'
                  ? '◎ No Robot · Scene Map Mode'
                  : '✕ Failure Detected'}
            </span>
            <span className="opacity-80">
              {detectedOutcome === 'success'
                ? 'Robustness Stress Test'
                : detectedOutcome === 'no_robot'
                  ? 'Sim-Ready Map'
                  : 'Policy Repair'}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-300">
            {session.sceneSpec.outcomeSummary}
          </p>
        </div>
      )}

      {/* Robot Kinematic Platform & Task Goal */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-slate-300">
            Target Robot Platform
          </label>
          <div className="grid grid-cols-2 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
            {(Object.keys(APP_CONFIG.robots) as RobotModelId[]).map((id) => (
              <button
                key={id}
                type="button"
                disabled={session.status === 'running'}
                onClick={() => {
                  logFunctionCall('CaptureIntakePanel.selectRobot', { robotModel: id });
                  onRobotModelChange(id);
                }}
                className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                  robotModel === id
                    ? 'bg-[#1E293B] text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {id === 'so_arm101' ? 'SO-ARM101' : 'Franka Panda'}
              </button>
            ))}
          </div>
        </div>

        {/* Simulation Rollout Duration Selector */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <label className="text-xs font-medium text-slate-300">
              Simulation Rollout Length
            </label>
            <span className="font-mono-tabular text-[10px] text-slate-400">
              {durationMode === 'match'
                ? detectedVideoDurationSec
                  ? `Clip ${Math.round(detectedVideoDurationSec)}s`
                  : `Auto (${APP_CONFIG.simulation.defaultRolloutDurationSec}s default)`
                : durationMode === 60
                  ? '1 minute (60s)'
                  : `${durationMode} seconds`}
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
            {(
              [
                {
                  id: 'match' as const,
                  label: detectedVideoDurationSec
                    ? `Clip (${Math.min(120, Math.max(3, Math.round(detectedVideoDurationSec)))}s)`
                    : 'Full Clip',
                },
                { id: 15 as const, label: '15s' },
                { id: 30 as const, label: '30s' },
                { id: 60 as const, label: '60s (1m)' },
              ] as const
            ).map((opt) => (
              <button
                key={String(opt.id)}
                type="button"
                disabled={session.status === 'running'}
                onClick={() => {
                  logFunctionCall('CaptureIntakePanel.selectRolloutDuration', {
                    durationMode: opt.id,
                  });
                  onDurationModeChange(opt.id);
                }}
                className={`px-1.5 py-1.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap truncate cursor-pointer ${
                  durationMode === opt.id
                    ? 'bg-[#1E293B] text-amber-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="task-goal-input" className="text-xs font-medium text-slate-300">
            Manipulation Task Goal
          </label>
          <textarea
            id="task-goal-input"
            rows={2}
            value={taskGoal}
            disabled={session.status === 'running'}
            onChange={(e) => onTaskGoalChange(e.target.value)}
            placeholder="Describe the grasp objective and collision constraints..."
            className="w-full px-3 py-2 text-xs text-slate-100 bg-[#0B0E17] border border-[#1E293B] rounded-md focus:outline-none focus:border-amber-500 resize-none"
          />
        </div>

        <button
          type="button"
          disabled={!canCompile}
          onClick={() => void onRunPipeline()}
          className={`relative overflow-hidden w-full py-2.5 px-4 inline-flex items-center justify-center gap-2 text-xs font-semibold rounded-md transition-colors ${
            pipelineProgress.isRunning
              ? 'bg-[#161D2C] border border-amber-500/45 text-amber-100 cursor-wait'
              : 'text-slate-950 bg-amber-500 hover:bg-amber-400 disabled:bg-[#1E293B] disabled:text-slate-500 cursor-pointer'
          }`}
        >
          {pipelineProgress.isRunning && (
            <span
              aria-hidden="true"
              className="absolute inset-y-0 left-0 bg-amber-500/25 transition-[width] duration-300 ease-out"
              style={{ width: `${pipelineProgress.percent}%` }}
            />
          )}
          <span className="relative z-10 inline-flex items-center justify-between w-full gap-2">
            <span className="inline-flex items-center justify-center gap-2 flex-1">
              <Play className="w-3.5 h-3.5 shrink-0" />
              <span>
                {pipelineProgress.isRunning
                  ? 'Analyzing & Verifying Policy...'
                  : session.iterations.length > 0
                    ? 'Re-Run Verification Pipeline'
                    : 'Analyze & Verify Policy'}
              </span>
            </span>
            {pipelineProgress.isRunning && (
              <span className="font-mono-tabular text-xs font-bold text-amber-300 shrink-0">
                {pipelineProgress.percent}%
              </span>
            )}
          </span>
        </button>

        {/* Numeric Pipeline Progress Bar & Elapsed Clock */}
        {(pipelineProgress.isRunning || pipelineProgress.isCompleted || pipelineProgress.isError) && (
          <div
            role="progressbar"
            aria-label="Policy verification progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pipelineProgress.percent}
            className="flex flex-col gap-1.5 bg-[#0B0E17] border border-[#1E293B] rounded-md px-3 py-2"
          >
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span
                className={`truncate font-medium ${
                  pipelineProgress.isCompleted
                    ? 'text-emerald-300'
                    : pipelineProgress.isError
                      ? 'text-rose-300'
                      : 'text-slate-200'
                }`}
                title={pipelineProgress.stageLabel}
              >
                {pipelineProgress.stageLabel}
              </span>
              <span
                className={`font-mono-tabular text-xs font-semibold shrink-0 ${
                  pipelineProgress.isCompleted
                    ? 'text-emerald-400'
                    : pipelineProgress.isError
                      ? 'text-rose-400'
                      : 'text-amber-400'
                }`}
              >
                {pipelineProgress.percent}%
              </span>
            </div>

            <div className="w-full h-1.5 bg-[#161D2C] rounded overflow-hidden">
              <div
                className={`h-full transition-[width] duration-300 ease-out ${
                  pipelineProgress.isCompleted
                    ? 'bg-emerald-500'
                    : pipelineProgress.isError
                      ? 'bg-rose-500'
                      : 'bg-amber-500'
                }`}
                style={{ width: `${pipelineProgress.percent}%` }}
              />
            </div>

            <div className="flex items-center justify-between gap-2 font-mono-tabular text-[10px] text-slate-400">
              <span>Elapsed {pipelineProgress.elapsedLabel}</span>
              <div className="flex items-center gap-2">
                <span>
                  {pipelineProgress.isRunning
                    ? APP_CONFIG.pipelineProgress.typicalTotalDurationLabel
                    : pipelineProgress.isCompleted
                      ? 'Complete'
                      : 'Halted'}
                </span>
                {pipelineProgress.isRunning && (
                  <button
                    type="button"
                    onClick={() => void onCancelPipeline()}
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium text-rose-300 bg-rose-500/10 border border-rose-500/35 rounded hover:bg-rose-500/20 hover:text-rose-200 transition-colors cursor-pointer"
                  >
                    <X className="w-2.5 h-2.5" />
                    <span>Cancel</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
