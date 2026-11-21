/**
 * Three.js WebGL Stage Video Clip Recorder (`stageVideoRecorder.ts`)
 *
 * Captures live real-time video clips directly from the interactive Three.js
 * WebGL canvas using `HTMLCanvasElement.captureStream()` and browser-native
 * `MediaRecorder` (VP9/VP8 WebM), allowing robotics engineers to record and
 * download arbitrary-length simulation clips — including full 60-second
 * (1-minute) or custom-duration rollouts in Single View, Dual Cam, or Dual
 * Arms split layouts.
 *
 * Use cases:
 * - Exporting a 15s, 30s, or 60s (1-minute) simulation clip of a repaired
 *   policy rollout or interactive live physics test for lab reports or demos.
 * - Stopping a recording early at any moment while still cleanly flushing and
 *   downloading the captured `.webm` video file.
 */

import { APP_CONFIG } from '../config/appConfig';
import { triggerBlobDownload } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

export interface StageClipRecorderHandle {
  /** Stops recording immediately and triggers download of the captured clip. */
  stop: () => void;
  /** Cancels recording without downloading a file (used on component unmount). */
  cancel: () => void;
}

export interface StartStageClipRecordingParams {
  /** Active Three.js WebGL canvas element to capture. */
  canvas: HTMLCanvasElement;
  /** Target recording duration in seconds (e.g. 15, 30, 60). */
  targetDurationSec: number;
  /** Base filename without extension (`.webm` is appended automatically). */
  filenameBase: string;
  /** Called periodically with elapsed recording seconds. */
  onTick: (elapsedSec: number) => void;
  /** Called once the clip has finished encoding and triggered download. */
  onComplete: (filename: string, sizeBytes: number, actualDurationSec: number) => void;
  /** Called if canvas capture or MediaRecorder is unsupported or fails. */
  onError: (message: string) => void;
}

/**
 * Picks the highest-quality WebM MediaRecorder MIME type supported by the
 * current browser.
 */
function pickSupportedWebmMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return '';
}

/**
 * Starts recording a live `.webm` video clip directly from the Three.js WebGL
 * canvas for up to `targetDurationSec` seconds (e.g., 60 seconds / 1 minute).
 */
export function startStageClipRecording(
  params: StartStageClipRecordingParams,
): StageClipRecorderHandle | null {
  logFunctionCall('startStageClipRecording', {
    targetDurationSec: params.targetDurationSec,
    filenameBase: params.filenameBase,
    fps: APP_CONFIG.simulation.videoExport.fps,
  });

  if (typeof MediaRecorder === 'undefined' || typeof params.canvas.captureStream !== 'function') {
    params.onError('In-browser WebGL video recording is not supported in this browser.');
    return null;
  }

  let stream: MediaStream;
  try {
    stream = params.canvas.captureStream(APP_CONFIG.simulation.videoExport.fps);
  } catch (err) {
    params.onError(
      err instanceof Error ? err.message : 'Unable to capture WebGL canvas stream.',
    );
    return null;
  }

  const mimeType = pickSupportedWebmMimeType();
  const options: MediaRecorderOptions = {
    videoBitsPerSecond: APP_CONFIG.simulation.videoExport.videoBitsPerSecond,
  };
  if (mimeType) {
    options.mimeType = mimeType;
  }

  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream, options);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    params.onError(
      err instanceof Error ? err.message : 'Unable to initialize MediaRecorder.',
    );
    return null;
  }

  const chunks: Blob[] = [];
  const startedAtMs = performance.now();
  let cancelled = false;
  let finished = false;

  const stopTracks = () => {
    stream.getTracks().forEach((track) => track.stop());
  };

  const tickInterval = window.setInterval(() => {
    const elapsedSec = (performance.now() - startedAtMs) / 1000;
    params.onTick(Math.min(params.targetDurationSec, elapsedSec));
    if (elapsedSec >= params.targetDurationSec && recorder.state === 'recording') {
      recorder.stop();
    }
  }, 200);

  recorder.ondataavailable = (evt) => {
    if (evt.data && evt.data.size > 0) {
      chunks.push(evt.data);
    }
  };

  recorder.onerror = () => {
    if (finished) return;
    finished = true;
    window.clearInterval(tickInterval);
    stopTracks();
    params.onError('WebGL video encoder encountered an error while recording.');
  };

  recorder.onstop = () => {
    if (finished) return;
    finished = true;
    window.clearInterval(tickInterval);
    stopTracks();

    if (cancelled) return;

    const actualDurationSec = Math.max(0.5, (performance.now() - startedAtMs) / 1000);
    const blob = new Blob(chunks, { type: mimeType || 'video/webm' });
    if (blob.size === 0) {
      params.onError('Recorded video clip was empty. Try keeping the stage tab active while recording.');
      return;
    }

    const safeBase = params.filenameBase.replace(/[^a-zA-Z0-9._-]/g, '_');
    const filename = `${safeBase}.webm`;
    triggerBlobDownload(blob, filename);
    params.onComplete(filename, blob.size, actualDurationSec);
  };

  recorder.start(500);
  params.onTick(0);

  return {
    stop: () => {
      if (recorder.state === 'recording') {
        recorder.stop();
      }
    },
    cancel: () => {
      cancelled = true;
      window.clearInterval(tickInterval);
      if (recorder.state === 'recording') {
        recorder.stop();
      } else {
        stopTracks();
      }
    },
  };
}
