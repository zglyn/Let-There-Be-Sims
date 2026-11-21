/**
 * Chunked Binary Video Upload Manager (`chunkedUploadManager.ts`)
 *
 * Manages proxy-safe, multi-part binary video ingestion over standard `fetch`
 * requests (`init` -> raw `application/octet-stream` chunks -> `complete`).
 * Splitting files into compact 1.5 MB binary chunks provides three critical
 * operational advantages in Cloud Run & proxied iframe environments:
 * 1. Works strictly over `window.fetch` (preserving preview-proxy cookies/auth
 *    and avoiding `XMLHttpRequest` redirect/interstitial edge cases).
 * 2. Bypasses Cloud Run's 32 MB HTTP/1.1 single-request payload cap so users
 *    can upload clips up to 60 MB reliably.
 * 3. Provides real per-chunk progress updates without multipart boundary parsing.
 *
 * All files are written strictly inside the confined temporary upload directory
 * with server-generated random filenames and validated MIME/size ceilings.
 *
 * Use cases:
 * - Uploading desktop `.mp4`/`.webm`/`.mov` robotics failure clips from Hugging
 *   Face datasets or local cameras.
 * - Streaming smartphone video recordings from `/capture?session=<id>`.
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { APP_CONFIG } from '../config/appConfig';
import type { UploadedVideoMeta, VideoSlot } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { resolveConfinedUploadPath } from './sessionStore';

interface PendingChunkUpload {
  uploadId: string;
  sessionId: string;
  slot: VideoSlot;
  filePath: string;
  originalName: string;
  mimeType: string;
  expectedSizeBytes: number;
  receivedBytes: number;
  source: UploadedVideoMeta['source'];
  createdAtMs: number;
}

const pendingUploads = new Map<string, PendingChunkUpload>();
const STALE_UPLOAD_TTL_MS = 15 * 60 * 1000;

/**
 * Infers a clean media MIME type from the uploaded filename extension whenever
 * a browser reports `application/octet-stream` or omits the MIME type.
 */
export function resolveUploadedMimeType(
  reportedMime: string | undefined,
  originalName: string,
): string {
  const baseMime = (reportedMime ?? '').split(';')[0].trim().toLowerCase();
  if (baseMime && baseMime !== 'application/octet-stream') {
    return baseMime;
  }
  const ext = path.extname(originalName).toLowerCase();
  const extMap: Record<string, string> = {
    '.mp4': 'video/mp4',
    '.m4v': 'video/mp4',
    '.webm': 'video/webm',
    '.mov': 'video/quicktime',
    '.3gp': 'video/3gpp',
    '.mkv': 'video/x-matroska',
    '.avi': 'video/x-msvideo',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
  };
  return extMap[ext] ?? baseMime;
}

/**
 * Validates whether a resolved MIME type is permitted by centralized upload rules.
 */
export function isAllowedMediaMime(mimeType: string): boolean {
  const allowed = APP_CONFIG.upload.allowedMimeTypes as readonly string[];
  return allowed.includes(mimeType) || mimeType.startsWith('video/');
}

/**
 * Purges abandoned chunked upload handles older than 15 minutes.
 */
function pruneStalePendingUploads(): void {
  const now = Date.now();
  for (const [id, entry] of pendingUploads.entries()) {
    if (now - entry.createdAtMs > STALE_UPLOAD_TTL_MS) {
      pendingUploads.delete(id);
      try {
        if (fs.existsSync(entry.filePath)) fs.unlinkSync(entry.filePath);
      } catch {
        // Ignore cleanup race errors
      }
    }
  }
}

/**
 * Initializes a new confined chunked upload handle on disk after validating
 * file type and byte size limits.
 */
export function initChunkedUpload(params: {
  sessionId: string;
  slot?: VideoSlot;
  filename: string;
  mimeType?: string;
  sizeBytes: number;
  source: UploadedVideoMeta['source'];
}): { uploadId: string; mimeType: string; slot: VideoSlot } {
  const slot: VideoSlot = params.slot === 'secondary' ? 'secondary' : 'primary';
  logFunctionCall('initChunkedUpload', {
    sessionId: params.sessionId,
    slot,
    filename: params.filename,
    mimeType: params.mimeType,
    sizeBytes: params.sizeBytes,
    source: params.source,
  });

  pruneStalePendingUploads();

  const originalName = (params.filename || 'capture.mp4').slice(0, 120);
  const resolvedMime = resolveUploadedMimeType(params.mimeType, originalName);
  if (!isAllowedMediaMime(resolvedMime)) {
    throw new Error(
      `Unsupported file type '${params.mimeType || path.extname(originalName)}'. Upload a video (MP4/WebM/MOV) or frame image.`,
    );
  }
  if (!Number.isFinite(params.sizeBytes) || params.sizeBytes <= 0) {
    throw new Error('Selected media file is empty (0 bytes).');
  }
  if (params.sizeBytes > APP_CONFIG.upload.maxFileSizeBytes) {
    const maxMb = Math.round(APP_CONFIG.upload.maxFileSizeBytes / (1024 * 1024));
    throw new Error(`File exceeds the ${maxMb} MB maximum upload limit.`);
  }

  const ext =
    path.extname(originalName).toLowerCase().replace(/[^.a-z0-9]/g, '') || '.mp4';
  const uploadId = `up_${crypto.randomBytes(8).toString('hex')}`;
  const safeFileName = `clip_${slot}_${Date.now()}_${crypto.randomBytes(6).toString('hex')}${ext}`;
  const confinedPath = resolveConfinedUploadPath(safeFileName);

  // Create empty confined file ready for sequential chunk appends
  fs.writeFileSync(confinedPath, Buffer.alloc(0));

  pendingUploads.set(uploadId, {
    uploadId,
    sessionId: params.sessionId,
    slot,
    filePath: confinedPath,
    originalName,
    mimeType: resolvedMime || 'video/mp4',
    expectedSizeBytes: params.sizeBytes,
    receivedBytes: 0,
    source: params.source,
    createdAtMs: Date.now(),
  });

  return { uploadId, mimeType: resolvedMime || 'video/mp4', slot };
}

/**
 * Appends a raw binary chunk to an initialized upload handle while enforcing
 * session ownership and cumulative file-size limits.
 */
export function appendUploadChunk(params: {
  sessionId: string;
  uploadId: string;
  chunkBuffer: Buffer;
  chunkIndex: number;
}): { receivedBytes: number } {
  logFunctionCall('appendUploadChunk', {
    sessionId: params.sessionId,
    uploadId: params.uploadId,
    chunkIndex: params.chunkIndex,
    chunkBytes: params.chunkBuffer.length,
  });

  const entry = pendingUploads.get(params.uploadId);
  if (!entry || entry.sessionId !== params.sessionId) {
    throw new Error('Upload session expired or invalid uploadId. Please retry uploading.');
  }

  const nextBytes = entry.receivedBytes + params.chunkBuffer.length;
  if (nextBytes > APP_CONFIG.upload.maxFileSizeBytes) {
    pendingUploads.delete(params.uploadId);
    try {
      if (fs.existsSync(entry.filePath)) fs.unlinkSync(entry.filePath);
    } catch {
      // Ignore cleanup race errors
    }
    throw new Error('Uploaded file exceeded maximum allowed size.');
  }

  fs.appendFileSync(entry.filePath, params.chunkBuffer);
  entry.receivedBytes = nextBytes;
  return { receivedBytes: nextBytes };
}

/**
 * Finalizes a completed chunked upload, verifies disk persistence, and returns
 * the confined file path plus populated `UploadedVideoMeta`.
 */
export function finalizeChunkedUpload(params: {
  sessionId: string;
  uploadId: string;
}): { filePath: string; videoMeta: UploadedVideoMeta; slot: VideoSlot } {
  logFunctionCall('finalizeChunkedUpload', {
    sessionId: params.sessionId,
    uploadId: params.uploadId,
  });

  const entry = pendingUploads.get(params.uploadId);
  if (!entry || entry.sessionId !== params.sessionId) {
    throw new Error('Upload handle not found or already finalized.');
  }
  pendingUploads.delete(params.uploadId);

  if (!fs.existsSync(entry.filePath)) {
    throw new Error('Uploaded file could not be located on server storage.');
  }
  const stat = fs.statSync(entry.filePath);
  if (stat.size <= 0) {
    throw new Error('Uploaded video contains 0 bytes.');
  }

  const baseMediaUrl = APP_CONFIG.api.mediaFile(params.sessionId, entry.slot);
  const sep = baseMediaUrl.includes('?') ? '&' : '?';

  const videoMeta: UploadedVideoMeta = {
    originalName: entry.originalName,
    mimeType: entry.mimeType,
    sizeBytes: stat.size,
    uploadedAt: new Date().toISOString(),
    source: entry.source,
    mediaUrl: `${baseMediaUrl}${sep}v=${Date.now()}`,
  };

  return { filePath: entry.filePath, videoMeta, slot: entry.slot };
}
