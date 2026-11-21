/**
 * Session & Telemetry Export Serializer (`sessionExporter.ts`)
 *
 * Converts an active `CompilerSessionState` into portable engineering artifacts
 * ready for external analysis, version control, or robotics hardware deployment:
 * 1. Complete JSON Session Snapshot (task goal, robot spec, Carbon spatial
 *    calibration, MJCF XML, 20-environment rollout metrics, policy patches, and
 *    Gemini 4 `previous_interaction_id` audit trail — excluding raw video blobs).
 * 2. Tabular CSV Telemetry Matrix (every iteration × 20 environments with
 *    contact force, lift height, slip velocity, and randomization parameters).
 * 3. Direct exports for the latest verified Python policy (`.py`) and calibrated
 *    MuJoCo scene (`.mjcf.xml`).
 *
 * Use cases:
 * - Exporting structured verification evidence for hackathon judging or CI logs.
 * - Loading 20-environment pass/fail telemetry into Pandas, Excel, or MATLAB.
 */

import type { CompilerSessionState } from '../types/robotics';
import { downloadTextArtifact } from './fileDownload';
import { logFunctionCall } from './logger';

/**
 * Escapes a single CSV cell according to RFC 4180 quoting rules.
 */
function escapeCsvCell(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (/[",\n\r]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Serializes the full compiler session state into a clean, indented JSON report
 * containing spatial calibration, rollout telemetry, and Gemini 4 interaction traces.
 */
export function buildSessionJsonExport(session: CompilerSessionState): string {
  logFunctionCall('buildSessionJsonExport', {
    sessionId: session.id,
    iterations: session.iterations.length,
    traces: session.interactionTrace.length,
  });

  const payload = {
    exportedAt: new Date().toISOString(),
    sessionId: session.id,
    createdAt: session.createdAt,
    status: session.status,
    taskGoal: session.taskGoal,
    robotModel: session.robotModel,
    video: session.video,
    webhook: session.webhook,
    stages: session.stages,
    sceneSpec: session.sceneSpec,
    iterations: session.iterations.map((iter) => ({
      iteration: iter.iteration,
      passed: iter.passed,
      total: iter.total,
      executionSource: iter.executionSource,
      telemetrySummary: iter.telemetrySummary,
      rootCauseFix: iter.rootCauseFix,
      controllerParams: iter.controllerParams,
      policyCode: iter.policyCode,
      hydrogenInteractionId: iter.hydrogenInteractionId,
      neonInteractionId: iter.neonInteractionId,
      rollouts: iter.rollouts.map((r) => ({
        envId: r.envId,
        success: r.success,
        failureReason: r.failureReason,
        finalObjectZ: r.finalObjectZ,
        peakContactForceN: r.peakContactForceN,
        maxSlipVelocityMs: r.maxSlipVelocityMs,
        params: r.params,
      })),
    })),
    interactionTrace: session.interactionTrace,
    bundleStatus: session.bundle
      ? {
          interactionId: session.bundle.interactionId,
          status: session.bundle.status,
          updatedAt: session.bundle.updatedAt,
        }
      : null,
  };

  return JSON.stringify(payload, null, 2);
}

/**
 * Builds a flat RFC 4180 CSV table spanning all executed iterations and all
 * 20 domain-randomized environments.
 */
export function buildTelemetryCsvExport(session: CompilerSessionState): string {
  logFunctionCall('buildTelemetryCsvExport', {
    sessionId: session.id,
    iterations: session.iterations.length,
  });

  const headers = [
    'session_id',
    'iteration',
    'env_id',
    'success',
    'peak_contact_force_n',
    'final_object_z_m',
    'max_slip_velocity_ms',
    'friction',
    'mass_kg',
    'xy_offset_x_m',
    'xy_offset_y_m',
    'failure_reason',
  ];

  const rows: string[] = [headers.join(',')];

  for (const iter of session.iterations) {
    for (const r of iter.rollouts) {
      rows.push(
        [
          session.id,
          iter.iteration,
          `E${String(r.envId).padStart(2, '0')}`,
          r.success,
          r.peakContactForceN,
          r.finalObjectZ,
          r.maxSlipVelocityMs,
          r.params.friction,
          r.params.mass,
          r.params.xyOffset[0],
          r.params.xyOffset[1],
          r.failureReason ?? '',
        ]
          .map(escapeCsvCell)
          .join(','),
      );
    }
  }

  return rows.join('\n');
}

/**
 * Downloads the complete session report as a JSON file.
 */
export function exportSessionJson(session: CompilerSessionState): void {
  logFunctionCall('exportSessionJson', { sessionId: session.id });
  const json = buildSessionJsonExport(session);
  downloadTextArtifact(
    `${session.id}_lettherebesims_report.json`,
    json,
    'application/json;charset=utf-8',
  );
}

/**
 * Downloads the 20-environment rollout telemetry matrix as a CSV spreadsheet.
 */
export function exportTelemetryCsv(session: CompilerSessionState): boolean {
  logFunctionCall('exportTelemetryCsv', { sessionId: session.id });
  if (session.iterations.length === 0) {
    return false;
  }
  const csv = buildTelemetryCsvExport(session);
  downloadTextArtifact(`${session.id}_swarm_telemetry.csv`, csv, 'text/csv;charset=utf-8');
  return true;
}

/**
 * Downloads the latest verified Python policy (`execute_policy`) if compiled.
 */
export function exportLatestPolicyPython(session: CompilerSessionState): boolean {
  logFunctionCall('exportLatestPolicyPython', { sessionId: session.id });
  const latestIter = session.iterations[session.iterations.length - 1];
  const code =
    session.bundle?.standaloneControllerPy ||
    latestIter?.policyCode ||
    session.sceneSpec?.initialControllerPython;
  if (!code) return false;
  downloadTextArtifact(`${session.id}_verified_policy.py`, code, 'text/x-python;charset=utf-8');
  return true;
}

/**
 * Downloads the calibrated MuJoCo MJCF XML scene if compiled by Gemini 4 Carbon.
 */
export function exportCompiledMjcfXml(session: CompilerSessionState): boolean {
  logFunctionCall('exportCompiledMjcfXml', { sessionId: session.id });
  const xml = session.bundle?.regressionSuiteXml || session.sceneSpec?.mjcfXml;
  if (!xml) return false;
  downloadTextArtifact(`${session.id}_scene.mjcf.xml`, xml, 'application/xml;charset=utf-8');
  return true;
}
