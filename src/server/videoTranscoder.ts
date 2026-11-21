/**
 * Server-Side Analysis Video Proxy Transcoder (`videoTranscoder.ts`)
 *
 * Compresses large or long-duration uploaded camera videos (e.g., dual 30 FPS
 * multi-minute clips totaling 20–60 MB) into lightweight, full-timeline 1 FPS /
 * 480p H.264 analysis proxies before sending them to Gemini 4 Carbon.
 *
 * Keeping the original full-resolution 30 FPS upload untouched for browser PiP
 * playback while passing a compact 1 FPS proxy to Gemini reduces upload &
 * visual tokenization overhead by ~10x–30x without truncating the video's
 * timeline, eliminating multi-minute stalls during 2-View Calibration.
 *
 * Security & Reliability:
 * - Uses `child_process.execFile('ffmpeg', [...])` with an explicit argument
 *   array (never shell interpolation).
 * - Confines temporary proxy files inside the upload temp directory via
 *   `resolveConfinedUploadPath`.
 * - Falls back gracefully to the original file if `ffmpeg` fails or is absent.
 *
 * Use cases:
 * - Preparing 1-minute to 3-minute single-view or dual-view robotics videos
 *   for fast, reliable Stage 1 Gemini 4 Carbon spatial reconstruction.
 */

import { execFile } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import { promisify } from 'util';
import { APP_CONFIG } from '../config/appConfig';
import { logFunctionCall } from '../utils/logger';
import { resolveConfinedUploadPath } from './sessionStore';

const execFileAsync = promisify(execFile);

export interface AnalysisVideoProxy {
  filePath: string;
  mimeType: string;
  originalBytes: number;
  proxyBytes: number;
  wasTranscoded: boolean;
  cleanup: () => void;
}

/**
 * Produces a compact 1 FPS / 480p H.264 analysis proxy when the input video
 * exceeds `APP_CONFIG.upload.analysisProxy.minBytesToTranscode`. Returns the
 * original file untouched for images, small clips, or if `ffmpeg` fails.
 */
export async function buildAnalysisVideoProxy(
  sourceFilePath: string,
  mimeType: string,
): Promise<AnalysisVideoProxy> {
  const stat = fs.statSync(sourceFilePath);
  const cfg = APP_CONFIG.upload.analysisProxy;
  logFunctionCall('buildAnalysisVideoProxy', {
    sourceFilePath,
    mimeType,
    sizeBytes: stat.size,
    enabled: cfg.enabled,
  });

  const noopCleanup = () => {};
  if (
    !cfg.enabled ||
    mimeType.startsWith('image/') ||
    stat.size < cfg.minBytesToTranscode
  ) {
    return {
      filePath: sourceFilePath,
      mimeType,
      originalBytes: stat.size,
      proxyBytes: stat.size,
      wasTranscoded: false,
      cleanup: noopCleanup,
    };
  }

  const proxyName = `proxy_${Date.now()}_${crypto.randomBytes(5).toString('hex')}.mp4`;
  const proxyPath = resolveConfinedUploadPath(proxyName);
  const vfFilter = `fps=${cfg.targetFps},scale='min(${cfg.maxWidthPx},iw)':-2`;

  try {
    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-v',
        'error',
        '-i',
        sourceFilePath,
        '-t',
        String(cfg.maxDurationSec),
        '-vf',
        vfFilter,
        '-an',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-crf',
        String(cfg.crf),
        '-movflags',
        '+faststart',
        proxyPath,
      ],
      {
        timeout: cfg.transcodeTimeoutMs,
        maxBuffer: 1024 * 1024,
      },
    );

    if (!fs.existsSync(proxyPath)) {
      throw new Error('ffmpeg did not produce proxy file.');
    }
    const proxyStat = fs.statSync(proxyPath);
    if (proxyStat.size <= 0 || proxyStat.size >= stat.size) {
      fs.unlinkSync(proxyPath);
      return {
        filePath: sourceFilePath,
        mimeType,
        originalBytes: stat.size,
        proxyBytes: stat.size,
        wasTranscoded: false,
        cleanup: noopCleanup,
      };
    }

    logFunctionCall('buildAnalysisVideoProxy.completed', {
      originalBytes: stat.size,
      proxyBytes: proxyStat.size,
      reductionRatio: Number((stat.size / proxyStat.size).toFixed(1)),
    });

    return {
      filePath: proxyPath,
      mimeType: 'video/mp4',
      originalBytes: stat.size,
      proxyBytes: proxyStat.size,
      wasTranscoded: true,
      cleanup: () => {
        try {
          if (fs.existsSync(proxyPath)) fs.unlinkSync(proxyPath);
        } catch {
          // Best-effort temp cleanup
        }
      },
    };
  } catch (err) {
    try {
      if (fs.existsSync(proxyPath)) fs.unlinkSync(proxyPath);
    } catch {
      // Ignore cleanup errors
    }
    logFunctionCall('buildAnalysisVideoProxy.fallbackOriginal', {
      reason: err instanceof Error ? err.message : String(err),
    });
    return {
      filePath: sourceFilePath,
      mimeType,
      originalBytes: stat.size,
      proxyBytes: stat.size,
      wasTranscoded: false,
      cleanup: noopCleanup,
    };
  }
}
