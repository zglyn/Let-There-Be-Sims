/**
 * Zero-Dependency Standard ZIP Archive Builder (`zipBuilder.ts`)
 *
 * Constructs RFC-compliant PKZIP (`.zip`) binary buffers directly in Node.js
 * using built-in `zlib.deflateRawSync` and IEEE 802.3 CRC-32 checksums.
 * Avoids extra runtime packaging dependencies while producing standard archives
 * that extract cleanly on macOS, Linux, Windows, Google Colab, and Modal.
 *
 * Use cases:
 * - Packaging multi-file Gymnasium + MuJoCo MJX RL training bundles as `.zip`.
 * - Packaging Hugging Face LeRobot v2.1 synthetic episode datasets (`meta/`,
 *   `data/chunk-000/`, `scripts/`, `scene.mjcf.xml`, `README.md`) as `.zip`.
 */

import zlib from 'zlib';
import { logFunctionCall } from './logger';

export interface ZipEntryInput {
  /** Relative POSIX path inside the ZIP archive (e.g. `meta/info.json`). */
  path: string;
  /** UTF-8 string or raw binary content to store in the archive. */
  content: string | Buffer;
}

/** Precomputed 256-entry IEEE 802.3 CRC-32 lookup table. */
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

/**
 * Computes the unsigned 32-bit CRC-32 checksum for a binary buffer.
 */
function computeCrc32(buf: Buffer): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc = CRC32_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Encodes a JavaScript Date into DOS time and DOS date 16-bit integers.
 */
function toDosDateTime(date: Date): { dosTime: number; dosDate: number } {
  const year = Math.max(1980, Math.min(2107, date.getUTCFullYear()));
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();
  const hours = date.getUTCHours();
  const minutes = date.getUTCMinutes();
  const seconds = Math.floor(date.getUTCSeconds() / 2);

  const dosTime = (hours << 11) | (minutes << 5) | seconds;
  const dosDate = ((year - 1980) << 9) | (month << 5) | day;
  return { dosTime, dosDate };
}

/**
 * Sanitizes an archive entry path to a clean relative POSIX path without
 * leading slashes or directory traversal segments.
 */
function normalizeZipPath(rawPath: string): string {
  return rawPath
    .replace(/\\/g, '/')
    .replace(/^\/+/, '')
    .split('/')
    .filter((seg) => seg.length > 0 && seg !== '.' && seg !== '..')
    .join('/');
}

/**
 * Builds a complete binary `.zip` archive buffer from an array of file entries
 * using DEFLATE compression (`method = 8`) with UTF-8 filename flags.
 */
export function buildZipArchive(entries: ZipEntryInput[]): Buffer {
  logFunctionCall('buildZipArchive', { entryCount: entries.length });

  const now = new Date();
  const { dosTime, dosDate } = toDosDateTime(now);
  const localParts: Buffer[] = [];
  const centralParts: Buffer[] = [];
  let offset = 0;

  for (const entry of entries) {
    const cleanPath = normalizeZipPath(entry.path);
    if (!cleanPath) continue;

    const nameBuf = Buffer.from(cleanPath, 'utf8');
    const rawBuf =
      typeof entry.content === 'string'
        ? Buffer.from(entry.content, 'utf8')
        : entry.content;
    const compressedBuf = zlib.deflateRawSync(rawBuf, { level: 6 });
    const crc = computeCrc32(rawBuf);

    // Local File Header (30 bytes + filename)
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0); // Local file header signature
    localHeader.writeUInt16LE(20, 4); // Version needed to extract (2.0)
    localHeader.writeUInt16LE(0x0800, 6); // General purpose bit flag: UTF-8 filename
    localHeader.writeUInt16LE(8, 8); // Compression method: DEFLATE
    localHeader.writeUInt16LE(dosTime, 10);
    localHeader.writeUInt16LE(dosDate, 12);
    localHeader.writeUInt32LE(crc, 14);
    localHeader.writeUInt32LE(compressedBuf.length, 18);
    localHeader.writeUInt32LE(rawBuf.length, 22);
    localHeader.writeUInt16LE(nameBuf.length, 26);
    localHeader.writeUInt16LE(0, 28); // Extra field length

    localParts.push(localHeader, nameBuf, compressedBuf);

    // Central Directory File Header (46 bytes + filename)
    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0); // Central directory signature
    centralHeader.writeUInt16LE(20, 4); // Version made by
    centralHeader.writeUInt16LE(20, 6); // Version needed
    centralHeader.writeUInt16LE(0x0800, 8); // UTF-8 filename flag
    centralHeader.writeUInt16LE(8, 10); // DEFLATE
    centralHeader.writeUInt16LE(dosTime, 12);
    centralHeader.writeUInt16LE(dosDate, 14);
    centralHeader.writeUInt32LE(crc, 16);
    centralHeader.writeUInt32LE(compressedBuf.length, 20);
    centralHeader.writeUInt32LE(rawBuf.length, 24);
    centralHeader.writeUInt16LE(nameBuf.length, 28);
    centralHeader.writeUInt16LE(0, 30); // Extra length
    centralHeader.writeUInt16LE(0, 32); // Comment length
    centralHeader.writeUInt16LE(0, 34); // Disk number start
    centralHeader.writeUInt16LE(0, 36); // Internal file attributes
    centralHeader.writeUInt32LE(0, 38); // External file attributes
    centralHeader.writeUInt32LE(offset, 42); // Relative offset of local header

    centralParts.push(centralHeader, nameBuf);

    offset += localHeader.length + nameBuf.length + compressedBuf.length;
  }

  const centralDirectoryBuf = Buffer.concat(centralParts);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); // End of central directory signature
  eocd.writeUInt16LE(0, 4); // Number of this disk
  eocd.writeUInt16LE(0, 6); // Disk where central directory starts
  eocd.writeUInt16LE(centralParts.length / 2, 8); // Entries on this disk
  eocd.writeUInt16LE(centralParts.length / 2, 10); // Total entries
  eocd.writeUInt32LE(centralDirectoryBuf.length, 12); // Size of central directory
  eocd.writeUInt32LE(offset, 16); // Offset of central directory
  eocd.writeUInt16LE(0, 20); // Comment length

  return Buffer.concat([...localParts, centralDirectoryBuf, eocd]);
}
