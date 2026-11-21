/**
 * Client-Side Artifact Download & Clipboard Helper
 *
 * Provides reliable, zero-popup browser downloads for compiled robotics
 * artifacts (standalone LeRobot/MuJoCo Python scripts, MJCF XML scenes, and
 * Markdown verification audit reports) without calling `window.open`.
 *
 * Use cases:
 * - Exporting verified deployment bundle files from the Bundle Inspector.
 * - Copying MJCF XML or Python policy snippets to clipboard in one click.
 */

import { logFunctionCall } from './logger';

/**
 * Triggers a browser download for any binary or text Blob using an ephemeral
 * object URL without calling `window.open`.
 */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  logFunctionCall('triggerBlobDownload', {
    filename,
    sizeBytes: blob.size,
    mimeType: blob.type,
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

/**
 * Triggers a browser download for a UTF-8 text artifact using an ephemeral
 * Blob object URL.
 */
export function downloadTextArtifact(filename: string, content: string, mimeType = 'text/plain;charset=utf-8'): void {
  logFunctionCall('downloadTextArtifact', { filename, contentLength: content.length, mimeType });
  const blob = new Blob([content], { type: mimeType });
  triggerBlobDownload(blob, filename);
}

/**
 * Copies text safely to the clipboard and returns true on success.
 */
export async function copyTextToClipboard(text: string): Promise<boolean> {
  logFunctionCall('copyTextToClipboard', { length: text.length });
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
