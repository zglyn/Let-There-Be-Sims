/**
 * Pipeline Progress Interpolation & Elapsed Telemetry Utility (`pipelineProgress.ts`)
 *
 * Computes smooth, monotonic, server-grounded progress percentages and formatted
 * elapsed timers for the 4-stage Gemini 4 verification pipeline. Rather than
 * stalling statically during 20–40 second reasoning turns (Gemini 4 Carbon or
 * Gemini 4 Neon `xhigh`) or fabricating fake 100% completion, this utility
 * combines authoritative backend milestone floors/ceilings (`PipelineProgressState`)
 * with a bounded exponential asymptote so progress visibly advances within each
 * active stage and snaps cleanly when each Gemini turn completes.
 *
 * Use cases:
 * - Powering the numeric progress bar directly on the "Analyzing & Verifying
 *   Policy..." button and status bar inside `CaptureIntakePanel`.
 * - Mirroring synchronized numeric percentage progress in `PipelineRail`.
 * - Formatting live elapsed `MM:SS` counters alongside expected stage windows.
 */

import { useEffect, useRef, useState } from 'react';
import { APP_CONFIG } from '../config/appConfig';
import type { CompilerSessionState, PipelineProgressState } from '../types/robotics';
import { logFunctionCall } from './logger';

export interface DisplayPipelineProgress {
  /** Integer 0–100 percentage safe for numeric UI readout and bar width. */
  percent: number;
  /** Human-readable description of the currently active pipeline step. */
  stageLabel: string;
  /** Formatted `MM:SS` elapsed duration since the pipeline run began. */
  elapsedLabel: string;
  /** Elapsed wall-clock seconds since pipeline start. */
  elapsedSeconds: number;
  /** Whether a pipeline run is actively executing. */
  isRunning: boolean;
  /** Whether the most recent pipeline run finished cleanly. */
  isCompleted: boolean;
  /** Whether the most recent pipeline run halted with an error. */
  isError: boolean;
}

/**
 * Formats a non-negative second count into a crisp `MM:SS` tabular clock string.
 */
export function formatElapsedSeconds(totalSeconds: number): string {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(clamped / 60);
  const secs = clamped % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Pure deterministic calculation of the current display percentage given the
 * server-broadcast `PipelineProgressState` and current wall-clock timestamp.
 *
 * Uses an asymptotic curve `floor + span * (1 - exp(-dt / expectedSec))` so
 * the indicator never exceeds `targetPercent - 1` until the backend confirms
 * completion of the active Gemini 4 turn.
 */
export function computeDisplayProgress(
  progress: PipelineProgressState | null,
  status: CompilerSessionState['status'],
  nowMs: number,
): number {
  if (status === 'completed') return 100;
  if (!progress) {
    return status === 'running' ? APP_CONFIG.pipelineProgress.mediaPrepStartPct : 0;
  }

  const floor = Math.max(0, Math.min(99, progress.percent));
  if (status !== 'running') {
    return Math.round(floor);
  }

  const ceiling = Math.max(floor, Math.min(99, progress.targetPercent - 1));
  const span = ceiling - floor;
  if (span <= 0) {
    return Math.round(floor);
  }

  const stageUpdatedMs = Date.parse(progress.stageUpdatedAt);
  const stageElapsedSec = Number.isFinite(stageUpdatedMs)
    ? Math.max(0, (nowMs - stageUpdatedMs) / 1000)
    : 0;
  const tau = Math.max(2, progress.expectedStageDurationSec);
  const eased = floor + span * (1 - Math.exp(-stageElapsedSec / tau));
  return Math.max(Math.round(floor), Math.min(Math.round(ceiling), Math.round(eased)));
}

/**
 * React hook subscribing to a lightweight timer while the pipeline is running
 * to deliver smooth monotonic percentage updates and live `MM:SS` elapsed time.
 */
export function usePipelineProgress(
  session: Pick<CompilerSessionState, 'id' | 'status' | 'progress' | 'stages'>,
): DisplayPipelineProgress {
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const highWaterRef = useRef<{ sessionId: string; startedAt: string | null; maxPct: number }>({
    sessionId: session.id,
    startedAt: session.progress?.startedAt ?? null,
    maxPct: 0,
  });

  const isRunning = session.status === 'running';
  const isCompleted = session.status === 'completed';
  const isError = session.status === 'error';

  useEffect(() => {
    if (!isRunning) {
      setNowMs(Date.now());
      return;
    }
    logFunctionCall('usePipelineProgress.startTicker', {
      sessionId: session.id,
      stageLabel: session.progress?.stageLabel ?? 'Initializing',
    });
    const timer = window.setInterval(() => {
      setNowMs(Date.now());
    }, APP_CONFIG.pipelineProgress.clientTickIntervalMs);
    return () => window.clearInterval(timer);
  }, [isRunning, session.id, session.progress?.stageLabel]);

  // Reset monotonic watermark whenever a new session or new pipeline run starts
  const currentStartedAt = session.progress?.startedAt ?? null;
  if (
    highWaterRef.current.sessionId !== session.id ||
    (currentStartedAt && highWaterRef.current.startedAt !== currentStartedAt) ||
    (!isRunning && !isCompleted && !isError)
  ) {
    highWaterRef.current = {
      sessionId: session.id,
      startedAt: currentStartedAt,
      maxPct: 0,
    };
  }

  const rawPercent = computeDisplayProgress(session.progress, session.status, nowMs);
  if (isCompleted) {
    highWaterRef.current.maxPct = 100;
  } else if (isRunning) {
    highWaterRef.current.maxPct = Math.max(highWaterRef.current.maxPct, rawPercent);
  } else if (isError) {
    highWaterRef.current.maxPct = Math.max(highWaterRef.current.maxPct, rawPercent);
  }

  const startedMs = session.progress?.startedAt ? Date.parse(session.progress.startedAt) : NaN;
  const elapsedSeconds =
    Number.isFinite(startedMs) && (isRunning || isCompleted || isError)
      ? Math.max(
          0,
          Math.floor(
            ((isRunning
              ? nowMs
              : Date.parse(session.progress?.stageUpdatedAt ?? '') || nowMs) -
              startedMs) /
              1000,
          ),
        )
      : 0;

  const activeStage = session.stages.find((s) => s.status === 'active');
  const stageLabel =
    session.progress?.stageLabel ||
    activeStage?.label ||
    (isCompleted
      ? 'Policy & Scene Verified'
      : isError
        ? 'Pipeline Halted'
        : 'Ready to Compile');

  return {
    percent: isCompleted ? 100 : highWaterRef.current.maxPct,
    stageLabel,
    elapsedLabel: formatElapsedSeconds(elapsedSeconds),
    elapsedSeconds,
    isRunning,
    isCompleted,
    isError,
  };
}
