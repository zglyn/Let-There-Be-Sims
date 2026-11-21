/**
 * Mobile-First QR-Paired Camera Capture Route (`MobileCaptureView`)
 *
 * Dedicated touch-ergonomic capture interface served at `/capture?session=<id>`
 * when scanned from the desktop command center's QR code (and also previewable
 * inline from the desktop top bar). Features:
 * - Touch targets >= 48px and clean dark viewfinder styling.
 * - Native rear-camera capture (`accept="video/*" capture="environment"`) for
 *   effortless iOS Safari / Android Chrome recording.
 * - Optional live in-browser `MediaRecorder` rear-camera viewfinder.
 * - Task goal confirmation, auto-compile toggle, live upload progress bar, and
 *   immediate SSE sync confirmation once sent to the paired desktop session.
 *
 * Use cases:
 * - Live hackathon demo: judge scans QR code on their phone, records 10s of a
 *   cluttered tabletop, and watches the big desktop screen compile the policy.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle2, Square, Upload, Video } from 'lucide-react';
import {
  fetchCompilerSession,
  uploadSessionVideo,
} from '../services/compilerApi';
import type { CompilerSessionState } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

interface MobileCaptureViewProps {
  sessionId: string;
  standalone: boolean;
  onUploadedToDesktop?: (session: CompilerSessionState) => void;
}

/**
 * Renders the mobile-first smartphone camera capture & direct cloud upload UI.
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

  const [session, setSession] = useState<CompilerSessionState | null>(null);
  const [taskGoal, setTaskGoal] = useState('');
  const [autoStart, setAutoStart] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadedOk, setUploadedOk] = useState(false);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    logFunctionCall('MobileCaptureView.hydrateSession', { sessionId });
    fetchCompilerSession(sessionId)
      .then((s) => {
        setSession(s);
        setTaskGoal(s.taskGoal);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Unable to pair with desktop session.');
      });
  }, [sessionId]);

  useEffect(() => {
    if (!isRecording) {
      setSeconds(0);
      return;
    }
    const id = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [isRecording]);

  const localPreviewRef = useRef<string | null>(null);

  useEffect(() => {
    localPreviewRef.current = localPreviewUrl;
  }, [localPreviewUrl]);

  useEffect(() => {
    return () => {
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
   * Streams the captured video blob or native camera file to the paired session.
   */
  const sendClipToSession = async (file: File | Blob, filename: string): Promise<void> => {
    logFunctionCall('MobileCaptureView.sendClipToSession', {
      sessionId,
      filename,
      sizeBytes: file.size,
      autoStart,
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
  };

  /**
   * Starts in-browser rear camera recording via MediaRecorder.
   */
  const handleStartLiveCamera = async (): Promise<void> => {
    logFunctionCall('MobileCaptureView.handleStartLiveCamera');
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Live MediaRecorder is unavailable; use the Native Rear Camera button above.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        await videoPreviewRef.current.play();
      }
      chunksRef.current = [];
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
        ? 'video/webm;codecs=vp8'
        : 'video/mp4';
      const recorder = new MediaRecorder(
        stream,
        MediaRecorder.isTypeSupported(mimeType) ? { mimeType } : undefined,
      );
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' });
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        setIsRecording(false);
        if (blob.size > 0) {
          await sendClipToSession(blob, `mobile_capture_${Date.now()}.webm`);
        }
      };

      recorder.start(250);
      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open rear camera.');
    }
  };

  /**
   * Stops live recording and uploads to the paired desktop session.
   */
  const handleStopLiveCamera = (): void => {
    logFunctionCall('MobileCaptureView.handleStopLiveCamera', { seconds });
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
  };

  return (
    <div
      className={
        standalone
          ? 'min-h-screen bg-[#07090E] text-slate-100 flex flex-col'
          : 'max-w-md mx-auto bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden'
      }
    >
      {/* Compact Header (<= 15% mobile sticky cap) */}
      <div className="px-4 py-3.5 bg-[#0B0E17] border-b border-[#1E293B] flex items-center justify-between gap-3">
        <div>
          <h1 className="font-display text-base font-bold text-slate-100">
            LetThereBeSims Mobile Capture
          </h1>
          <p className="font-mono-tabular text-[11px] text-cyan-400">
            Paired Session · {sessionId}
          </p>
        </div>
        <span className="font-mono-tabular text-[11px] text-emerald-400">
          {session ? '✓ Linked' : 'Connecting'}
        </span>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1">
        {error && (
          <div className="p-3 rounded-md bg-rose-500/10 border border-rose-500/35 text-xs text-rose-200">
            {error}
          </div>
        )}

        {uploadedOk && (
          <div className="p-3.5 rounded-md bg-emerald-500/12 border border-emerald-500/35 flex items-start gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div className="text-xs text-emerald-100 leading-relaxed">
              <p className="font-semibold">Footage Streamed to Desktop Command Center</p>
              <p className="text-emerald-200/80 mt-0.5">
                {autoStart
                  ? 'Gemini 4 Carbon spatial reconstruction has started automatically on the desktop screen.'
                  : 'Ready on the desktop dashboard. Tap Compile Verified Policy to launch.'}
              </p>
            </div>
          </div>
        )}

        {localPreviewUrl && (
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg p-2.5 flex flex-col gap-1.5">
            <span className="text-[11px] font-medium text-slate-300">Captured Footage Preview</span>
            <video
              key={localPreviewUrl}
              src={localPreviewUrl}
              controls
              muted
              playsInline
              preload="auto"
              className="w-full h-40 object-contain rounded bg-black border border-[#1E293B]"
            />
          </div>
        )}

        {/* Native Rear Camera Input (Recommended for iOS Safari / Android Chrome) */}
        <input
          ref={nativeCameraRef}
          type="file"
          accept="video/*,.mp4,.webm,.mov"
          capture="environment"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) void sendClipToSession(file, file.name);
          }}
          className="hidden"
        />

        <button
          type="button"
          disabled={uploadProgress !== null}
          onClick={() => nativeCameraRef.current?.click()}
          className="w-full min-h-[54px] px-4 py-3.5 inline-flex items-center justify-center gap-2.5 text-sm font-semibold text-slate-950 bg-amber-500 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors cursor-pointer"
        >
          <Camera className="w-5 h-5" />
          <span>Record with Rear Camera</span>
        </button>

        {/* Live Viewfinder MediaRecorder Fallback */}
        <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg p-3.5 flex flex-col gap-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-200">In-Browser Live Recorder</span>
            {isRecording && (
              <span className="font-mono-tabular text-rose-400 font-semibold">
                REC 00:{String(seconds).padStart(2, '0')}
              </span>
            )}
          </div>

          <video
            ref={videoPreviewRef}
            muted
            playsInline
            className={`w-full h-48 object-cover rounded-md bg-black border border-[#1E293B] ${
              isRecording ? 'block' : 'hidden'
            }`}
          />

          {isRecording ? (
            <button
              type="button"
              onClick={handleStopLiveCamera}
              className="w-full min-h-[48px] px-4 py-2.5 inline-flex items-center justify-center gap-2 text-sm font-semibold text-white bg-rose-600 rounded-md hover:bg-rose-500 transition-colors cursor-pointer"
            >
              <Square className="w-4 h-4" />
              <span>Stop & Stream to Cloud</span>
            </button>
          ) : (
            <button
              type="button"
              disabled={uploadProgress !== null}
              onClick={() => void handleStartLiveCamera()}
              className="w-full min-h-[48px] px-4 py-2.5 inline-flex items-center justify-center gap-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded-md hover:border-[#2D3B52] disabled:opacity-50 transition-colors cursor-pointer"
            >
              <Video className="w-4 h-4" />
              <span>Open Live Viewfinder</span>
            </button>
          )}
        </div>

        {/* Upload Progress Bar */}
        {uploadProgress !== null && (
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-lg p-3.5 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-mono-tabular text-amber-300">
              <span className="inline-flex items-center gap-1.5">
                <Upload className="w-3.5 h-3.5" />
                Uploading to paired session...
              </span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full h-2 bg-[#161D2C] rounded overflow-hidden">
              <div
                className="h-full bg-amber-500 transition-transform"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Task Goal & Auto-Compile Toggle */}
        <div className="flex flex-col gap-3 pt-1">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="mobile-goal" className="text-xs font-medium text-slate-300">
              Task Goal
            </label>
            <textarea
              id="mobile-goal"
              rows={2}
              value={taskGoal}
              onChange={(e) => setTaskGoal(e.target.value)}
              className="w-full px-3 py-2.5 text-sm text-slate-100 bg-[#0B0E17] border border-[#1E293B] rounded-md focus:outline-none focus:border-amber-500 resize-none"
            />
          </div>

          <label className="flex items-center justify-between gap-3 p-3 bg-[#0B0E17] border border-[#1E293B] rounded-md cursor-pointer min-h-[48px]">
            <span className="text-xs text-slate-200">
              Auto-start Gemini 4 pipeline after upload
            </span>
            <input
              type="checkbox"
              checked={autoStart}
              onChange={(e) => setAutoStart(e.target.checked)}
              className="w-5 h-5 accent-amber-500"
            />
          </label>
        </div>
      </div>
    </div>
  );
};
