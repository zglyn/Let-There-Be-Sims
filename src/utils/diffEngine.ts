/**
 * Line-by-Line Python Policy Diff Engine
 *
 * Computes a clean longest-common-subsequence (LCS) line diff between two
 * versions of `execute_policy(model, data, step)` generated across consecutive
 * Gemini 4 Neon (`xhigh`) repair iterations.
 *
 * Use cases:
 * - Highlighting exact waypoint, impedance gain (`kp`/`kd`), approach height,
 *   and grip force ramp edits made by Gemini 4 Neon in the Policy Diff view.
 * - Calculating added/removed line counts for crisp tabular audit summaries.
 */

import { logFunctionCall } from './logger';

export interface DiffLine {
  type: 'context' | 'added' | 'removed';
  oldLineNumber: number | null;
  newLineNumber: number | null;
  text: string;
}

export interface PolicyDiffResult {
  lines: DiffLine[];
  addedCount: number;
  removedCount: number;
}

/**
 * Computes a line-level LCS diff between `previousCode` and `currentCode`.
 */
export function computePolicyDiff(previousCode: string | null, currentCode: string): PolicyDiffResult {
  logFunctionCall('computePolicyDiff', {
    previousLength: previousCode?.length ?? 0,
    currentLength: currentCode.length,
  });

  const newLines = currentCode.split('\n');
  if (!previousCode) {
    return {
      lines: newLines.map((text, idx) => ({
        type: 'added',
        oldLineNumber: null,
        newLineNumber: idx + 1,
        text,
      })),
      addedCount: newLines.length,
      removedCount: 0,
    };
  }

  const oldLines = previousCode.split('\n');
  const m = oldLines.length;
  const n = newLines.length;

  // Guard LCS table size for very large scripts
  if (m * n > 200_000) {
    return {
      lines: [
        ...oldLines.map((text, i) => ({
          type: 'removed' as const,
          oldLineNumber: i + 1,
          newLineNumber: null,
          text,
        })),
        ...newLines.map((text, i) => ({
          type: 'added' as const,
          oldLineNumber: null,
          newLineNumber: i + 1,
          text,
        })),
      ],
      addedCount: n,
      removedCount: m,
    };
  }

  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (oldLines[i] === newLines[j]) {
        dp[i][j] = dp[i + 1][j + 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  const lines: DiffLine[] = [];
  let i = 0;
  let j = 0;
  let addedCount = 0;
  let removedCount = 0;

  while (i < m && j < n) {
    if (oldLines[i] === newLines[j]) {
      lines.push({
        type: 'context',
        oldLineNumber: i + 1,
        newLineNumber: j + 1,
        text: oldLines[i],
      });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      lines.push({
        type: 'removed',
        oldLineNumber: i + 1,
        newLineNumber: null,
        text: oldLines[i],
      });
      removedCount++;
      i++;
    } else {
      lines.push({
        type: 'added',
        oldLineNumber: null,
        newLineNumber: j + 1,
        text: newLines[j],
      });
      addedCount++;
      j++;
    }
  }

  while (i < m) {
    lines.push({
      type: 'removed',
      oldLineNumber: i + 1,
      newLineNumber: null,
      text: oldLines[i],
    });
    removedCount++;
    i++;
  }

  while (j < n) {
    lines.push({
      type: 'added',
      oldLineNumber: null,
      newLineNumber: j + 1,
      text: newLines[j],
    });
    addedCount++;
    j++;
  }

  return { lines, addedCount, removedCount };
}
