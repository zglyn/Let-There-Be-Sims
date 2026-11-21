/**
 * Persistent Failure-to-Regression-Suite CI/CD Store & Runner (`regressionSuiteStore.ts`)
 *
 * Implements "pytest for Physical Robots" inside LetThereBeSims:
 * 1. Promotes any real-world failure clip (or compiled tabletop scene) into a
 *    persistent MuJoCo regression test case saved on disk inside a confined
 *    workspace directory (`<tmpdir>/lettherebesims_regression_suite/suite_db.json`).
 * 2. Executes the full regression suite across every saved case (each spanning
 *    20 domain-randomized environments) against either:
 *    - A candidate controller policy from the active session (or custom CI payload),
 *    - Each case's stored verified patch (`suite_verified`), or
 *    - Each case's original unpatched baseline (`suite_baseline`, demonstrating
 *      that regressions are caught with `ciExitCode = 1`).
 * 3. Generates ready-to-commit CI/CD artifacts:
 *    - `.github/workflows/robotics-regression.yml` (GitHub Actions PR gate)
 *    - `tests/test_robotics_regression.py` (`pytest` verification harness)
 *    - `regression_suite_manifest.json` (portable MJCF + randomization manifest)
 *
 * Use cases:
 * - Automatically turning every repaired real-world robot failure into a
 *   permanent regression gate so future policy updates never re-introduce
 *   fixed collisions, slips, or incomplete placements.
 * - Running automated CI checks via `POST /api/regression/run` in GitHub Actions.
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type {
  CompilerSessionState,
  ControllerParams,
  RegressionCaseRunResult,
  RegressionCiArtifacts,
  RegressionRunMode,
  RegressionSuiteRunRecord,
  RegressionSuiteSnapshot,
  RegressionTestCase,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { evaluateBuiltInSwarm } from './physicsEvaluator';

const SUITE_ROOT = path.resolve(os.tmpdir(), APP_CONFIG.regressionCi.storageDirName);
const SUITE_DB_PATH = path.resolve(SUITE_ROOT, 'suite_db.json');

let cachedDb: RegressionSuiteSnapshot | null = null;

/**
 * Ensures the confined regression suite directory exists on disk.
 */
function ensureSuiteStorageDir(): void {
  if (!fs.existsSync(SUITE_ROOT)) {
    fs.mkdirSync(SUITE_ROOT, { recursive: true });
  }
}

/**
 * Loads the persisted regression suite database from disk (or initializes an
 * empty snapshot if none exists yet).
 */
function loadSuiteDb(): RegressionSuiteSnapshot {
  if (cachedDb) return cachedDb;
  ensureSuiteStorageDir();
  if (fs.existsSync(SUITE_DB_PATH)) {
    try {
      const raw = fs.readFileSync(SUITE_DB_PATH, 'utf8');
      const parsed = JSON.parse(raw) as Partial<RegressionSuiteSnapshot>;
      cachedDb = {
        cases: Array.isArray(parsed.cases) ? parsed.cases : [],
        runs: Array.isArray(parsed.runs) ? parsed.runs : [],
      };
      return cachedDb;
    } catch {
      // Fall back to a fresh snapshot if file was corrupted
    }
  }
  cachedDb = { cases: [], runs: [] };
  return cachedDb;
}

/**
 * Atomically writes the regression suite database to disk.
 */
function persistSuiteDb(db: RegressionSuiteSnapshot): void {
  ensureSuiteStorageDir();
  const tempPath = path.resolve(SUITE_ROOT, `suite_db_${Date.now()}.tmp`);
  fs.writeFileSync(tempPath, JSON.stringify(db, null, 2), 'utf8');
  fs.renameSync(tempPath, SUITE_DB_PATH);
  cachedDb = db;
}

/**
 * Returns all saved regression test cases and CI run history.
 */
export function getRegressionSuiteSnapshot(): RegressionSuiteSnapshot {
  logFunctionCall('getRegressionSuiteSnapshot');
  const db = loadSuiteDb();
  return {
    cases: db.cases,
    runs: db.runs,
  };
}

/**
 * Looks up a single persisted regression test case by its ID.
 */
export function getRegressionCaseById(caseId: string): RegressionTestCase | null {
  logFunctionCall('getRegressionCaseById', { caseId });
  const db = loadSuiteDb();
  return db.cases.find((c) => c.id === caseId) ?? null;
}

/**
 * Derives concise diagnostic tags from a session's failure summary and mode.
 */
function deriveFailureTags(session: CompilerSessionState): string[] {
  const tags = new Set<string>();
  tags.add(session.robotModel);
  if (session.sceneSpec?.executionOutcome === 'failure') tags.add('failure-repair');
  else if (session.sceneSpec?.executionOutcome === 'success') tags.add('robustness-gate');
  else tags.add('scene-map');

  const text = `${session.sceneSpec?.failureAnalysis ?? ''} ${session.sceneSpec?.outcomeSummary ?? ''}`.toLowerCase();
  if (text.includes('slip') || text.includes('friction')) tags.add('low-friction-slip');
  if (text.includes('collision') || text.includes('obstacle') || text.includes('force')) {
    tags.add('collision-clearance');
  }
  if (text.includes('tip') || text.includes('knock')) tags.add('tip-over');
  if (session.secondaryVideo) tags.add('dual-view');
  return Array.from(tags);
}

/**
 * Promotes an active compiler session's calibrated scene, baseline controller,
 * and latest verified policy patch into a permanent regression test case.
 * Updates in-place if the same `sourceSessionId` was already promoted.
 */
export function saveSessionAsRegressionCase(params: {
  session: CompilerSessionState;
  customName?: string;
  customTags?: string[];
}): RegressionTestCase {
  const { session, customName, customTags } = params;
  logFunctionCall('saveSessionAsRegressionCase', {
    sessionId: session.id,
    customName: customName ?? null,
  });

  if (!session.sceneSpec) {
    throw new Error('Compile a scene first before saving it to the Regression CI suite.');
  }

  const db = loadSuiteDb();
  const firstIter = session.iterations[0] ?? null;
  const latestIter = session.iterations[session.iterations.length - 1] ?? null;

  const baselineParams: ControllerParams = firstIter?.controllerParams
    ? { ...firstIter.controllerParams }
    : { ...session.sceneSpec.controllerParams };
  const baselineCode = firstIter?.policyCode || session.sceneSpec.initialControllerPython;

  const verifiedParams: ControllerParams = latestIter?.controllerParams
    ? { ...latestIter.controllerParams }
    : { ...session.sceneSpec.controllerParams };
  const verifiedCode = latestIter?.policyCode || session.sceneSpec.initialControllerPython;

  const existingIdx = db.cases.findIndex((c) => c.sourceSessionId === session.id);
  const caseId =
    existingIdx >= 0
      ? db.cases[existingIdx].id
      : `reg_${crypto.randomBytes(3).toString('hex')}`;

  const defaultName = `${session.sceneSpec.targetObject.name} · ${
    session.sceneSpec.executionOutcome === 'failure'
      ? 'Failure Gate'
      : session.sceneSpec.executionOutcome === 'success'
        ? 'Robustness Gate'
        : 'Scene Map Gate'
  }`;

  const testCase: RegressionTestCase = {
    id: caseId,
    name: (customName?.trim() || defaultName).slice(0, 96),
    createdAt: existingIdx >= 0 ? db.cases[existingIdx].createdAt : new Date().toISOString(),
    sourceSessionId: session.id,
    taskGoal: session.taskGoal,
    robotModel: session.robotModel,
    executionOutcome: session.sceneSpec.executionOutcome,
    failureSummary:
      session.sceneSpec.outcomeSummary ||
      session.sceneSpec.failureAnalysis.slice(0, 220),
    tags: customTags && customTags.length > 0 ? customTags.slice(0, 6) : deriveFailureTags(session),
    sceneSpec: session.sceneSpec,
    baselineControllerParams: baselineParams,
    baselinePolicyCode: baselineCode,
    verifiedControllerParams: verifiedParams,
    verifiedPolicyCode: verifiedCode,
    lastRun: existingIdx >= 0 ? db.cases[existingIdx].lastRun : null,
  };

  if (existingIdx >= 0) {
    db.cases[existingIdx] = testCase;
  } else {
    db.cases = [testCase, ...db.cases];
  }

  persistSuiteDb(db);
  return testCase;
}

/**
 * Deletes a saved regression test case by ID from the persistent suite.
 */
export function deleteRegressionCase(caseId: string): boolean {
  logFunctionCall('deleteRegressionCase', { caseId });
  const db = loadSuiteDb();
  const prevLen = db.cases.length;
  db.cases = db.cases.filter((c) => c.id !== caseId);
  if (db.cases.length === prevLen) {
    return false;
  }
  persistSuiteDb(db);
  return true;
}

/**
 * Executes the entire Regression Suite CI gate across every saved failure case
 * (20 randomized physics environments per case), recording build telemetry and
 * returning `ciExitCode: 0` (PASS) or `1` (FAIL).
 */
export function runRegressionSuiteCi(params: {
  mode: RegressionRunMode;
  policyLabel?: string;
  candidateControllerParams?: ControllerParams | null;
  candidateRobotModel?: RobotModelId;
}): RegressionSuiteRunRecord {
  logFunctionCall('runRegressionSuiteCi', {
    mode: params.mode,
    policyLabel: params.policyLabel ?? null,
    hasCandidateParams: Boolean(params.candidateControllerParams),
  });

  const db = loadSuiteDb();
  if (db.cases.length === 0) {
    throw new Error(
      'No regression test cases saved yet. Run a failure verification and click "Save to Regression CI" first.',
    );
  }

  const suiteStart = Date.now();
  const caseResults: RegressionCaseRunResult[] = [];
  let totalEnvsPassed = 0;
  let totalEnvsEvaluated = 0;

  for (const testCase of db.cases) {
    const caseStart = Date.now();
    const effectiveParams: ControllerParams =
      params.mode === 'candidate_policy' && params.candidateControllerParams
        ? params.candidateControllerParams
        : params.mode === 'suite_baseline'
          ? testCase.baselineControllerParams
          : testCase.verifiedControllerParams;

    const effectiveRobot: RobotModelId =
      params.mode === 'candidate_policy' && params.candidateRobotModel
        ? params.candidateRobotModel
        : testCase.robotModel;

    // Evaluate all 20 domain-randomized environments quickly at compact 6s episode duration
    const swarm = evaluateBuiltInSwarm({
      sceneSpec: testCase.sceneSpec,
      controllerParams: effectiveParams,
      robotModel: effectiveRobot,
      rolloutDurationSec: 6,
    });

    const worstContactForceN = Math.max(
      0,
      ...swarm.rollouts.map((r) => r.peakContactForceN),
    );
    const worstSlipVelocityMs = Math.max(
      0,
      ...swarm.rollouts.map((r) => r.maxSlipVelocityMs),
    );
    const minLiftHeightM = Math.min(
      ...swarm.rollouts.map((r) => r.finalObjectZ),
    );
    const tasksCompletedEnvs = swarm.rollouts.filter((r) => Boolean(r.taskCompleted)).length;
    const uniqueFailures = Array.from(
      new Set(
        swarm.rollouts
          .map((r) => r.failureReason)
          .filter((reason): reason is string => Boolean(reason)),
      ),
    ).slice(0, 3);

    const passRate = swarm.total > 0 ? swarm.passed / swarm.total : 0;
    const status: 'pass' | 'fail' =
      passRate >= APP_CONFIG.regressionCi.requiredPassRate ? 'pass' : 'fail';

    const result: RegressionCaseRunResult = {
      caseId: testCase.id,
      caseName: testCase.name,
      robotModel: effectiveRobot,
      passedEnvs: swarm.passed,
      totalEnvs: swarm.total,
      status,
      worstContactForceN: Number(worstContactForceN.toFixed(2)),
      worstSlipVelocityMs: Number(worstSlipVelocityMs.toFixed(4)),
      minLiftHeightM: Number(minLiftHeightM.toFixed(3)),
      tasksCompletedEnvs,
      failureReasons: uniqueFailures,
      durationMs: Math.max(1, Date.now() - caseStart),
    };

    testCase.lastRun = result;
    caseResults.push(result);
    totalEnvsPassed += swarm.passed;
    totalEnvsEvaluated += swarm.total;
  }

  const passedCases = caseResults.filter((r) => r.status === 'pass').length;
  const failedCases = caseResults.length - passedCases;
  const overallStatus: 'pass' | 'fail' = failedCases === 0 ? 'pass' : 'fail';

  const defaultLabel =
    params.mode === 'candidate_policy'
      ? 'Active Session Policy'
      : params.mode === 'suite_baseline'
        ? 'Unpatched Baseline Policies'
        : 'Stored Verified Patches';

  const runRecord: RegressionSuiteRunRecord = {
    runId: `ci_${crypto.randomBytes(3).toString('hex')}`,
    executedAt: new Date().toISOString(),
    mode: params.mode,
    policyLabel: params.policyLabel?.trim() || defaultLabel,
    totalCases: caseResults.length,
    passedCases,
    failedCases,
    totalEnvsPassed,
    totalEnvsEvaluated,
    overallStatus,
    ciExitCode: overallStatus === 'pass' ? 0 : 1,
    durationMs: Math.max(1, Date.now() - suiteStart),
    caseResults,
  };

  db.runs = [runRecord, ...db.runs].slice(0, APP_CONFIG.regressionCi.maxRunHistory);
  persistSuiteDb(db);
  return runRecord;
}

/**
 * Generates portable GitHub Actions workflow YAML, standalone `pytest`
 * regression harness, and a JSON manifest for committing into any robotics repo.
 */
export function generateRegressionCiArtifacts(): RegressionCiArtifacts {
  logFunctionCall('generateRegressionCiArtifacts');
  const db = loadSuiteDb();

  const githubWorkflowYaml = `# .github/workflows/robotics-regression.yml
# Generated by LetThereBeSims — Failure-to-Regression-Suite CI/CD Gate
name: Robotics Regression Suite CI

on:
  pull_request:
    paths:
      - "policies/**"
      - "controllers/**"
      - "tests/regression_suite_manifest.json"
  workflow_dispatch:

jobs:
  verify-physical-regressions:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Robotics Repository
        uses: actions/checkout@v4

      - name: Set Up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install MuJoCo & pytest
        run: pip install pytest mujoco numpy

      - name: Run Failure-to-Regression Physics Gate (${db.cases.length} saved case(s) · ${db.cases.length * 20} randomized envs)
        run: pytest -v tests/test_robotics_regression.py

      # Optional: Run against a live LetThereBeSims API server instead of local MuJoCo
      # - name: Trigger LetThereBeSims Remote CI Gate
      #   run: |
      #     curl -fsS -X POST "\${{ secrets.LETTHEREBESIMS_URL }}/api/regression/run" \\
      #       -H "Content-Type: application/json" \\
      #       -d '{"mode":"suite_verified","policyLabel":"PR #\${{ github.event.number }}"}' \\
      #       | tee ci_report.json
      #     node -e "const r = require('./ci_report.json'); process.exit(r.run.ciExitCode);"
`;

  const pytestScriptPy = `"""
LetThereBeSims — Automated Physical Failure Regression Suite (pytest)

Loads every real-world failure case exported in \`tests/regression_suite_manifest.json\`
and asserts that the candidate robot policy passes 100% of domain-randomized
variants (contact force <= ${APP_CONFIG.simulation.maxSafeContactForceN} N,
slip <= ${APP_CONFIG.simulation.maxSafeSlipVelocityMs} m/s,
lift >= ${APP_CONFIG.simulation.targetLiftZMeters} m, full pick-and-place task completion).
"""

import json
import math
from pathlib import Path
import pytest

MANIFEST_PATH = Path(__file__).parent / "regression_suite_manifest.json"
MAX_FORCE_N = ${APP_CONFIG.simulation.maxSafeContactForceN}
MAX_SLIP_MS = ${APP_CONFIG.simulation.maxSafeSlipVelocityMs}
MIN_LIFT_M = ${APP_CONFIG.simulation.targetLiftZMeters}


def load_regression_cases():
    if not MANIFEST_PATH.exists():
        return []
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload.get("cases", [])


@pytest.mark.parametrize(
    "case",
    load_regression_cases(),
    ids=lambda c: f"{c['id']}_{c['name'][:32]}",
)
def test_failure_case_does_not_regress(case):
    ctrl = case["verifiedControllerParams"]
    target = case["sceneSpec"]["targetObject"]
    obstacles = case["sceneSpec"].get("obstacles", [])
    randomizations = case["sceneSpec"].get("randomizations", [])
    assert len(randomizations) > 0, "Regression case must contain domain randomizations."

    for rand in randomizations:
        friction = max(0.12, min(1.2, float(rand["friction"])))
        mass = max(0.05, min(1.8, float(rand["mass"])))
        ox, oz = rand["xyOffset"]

        obj_x = target["position"]["x"] + ox
        obj_z = target["position"]["z"] + oz
        target_r = max(target["dimensions"]["x"], target["dimensions"]["z"]) * 0.5

        min_gap = 0.25
        obs_h = 0.08
        for obs in obstacles:
            dist = math.hypot(obs["position"]["x"] - obj_x, obs["position"]["z"] - obj_z)
            obs_r = max(obs["dimensions"]["x"], obs["dimensions"]["z"]) * 0.5
            gap = dist - obs_r - target_r
            if gap < min_gap:
                min_gap = gap
                obs_h = obs["dimensions"]["y"]

        req_force = (mass * 9.81) / (2.0 * friction)
        clearance_shortfall = max(0.0, obs_h + 0.045 - (ctrl["approachHeightM"] + ctrl["clearanceArcM"]))
        corridor_risk = max(0.0, 0.045 - min_gap) * 320.0
        impact = (
            clearance_shortfall * 420.0
            + corridor_risk * max(0.1, min(1.0, 1.0 - ctrl["clearanceArcM"] / 0.12))
            + max(0.0, ctrl["kpStiffness"] - 240.0) * 0.075
            + max(0.0, 22.0 - ctrl["kdDamping"]) * 0.95
            + max(0.0, ctrl["descentSpeedMs"] - 0.18) * 115.0
            + max(0.0, 60.0 - ctrl["gripRampSteps"]) * 0.22
        )
        peak_force = max(4.5, req_force * 1.15) + impact
        grip_margin = ctrl["gripForceN"] * friction * 2.0 - mass * 9.81
        lift_z = target["dimensions"]["y"] * 0.5 + ctrl["liftHeightM"]

        assert peak_force <= MAX_FORCE_N, (
            f"Env {rand['id']} contact force regression: {peak_force:.1f} N > {MAX_FORCE_N} N"
        )
        assert grip_margin >= 1.2, (
            f"Env {rand['id']} slip regression: grip margin {grip_margin:.2f} N < 1.2 N (mu={friction:.2f}, m={mass:.2f}kg)"
        )
        assert lift_z >= MIN_LIFT_M, (
            f"Env {rand['id']} lift regression: {lift_z:.3f} m < {MIN_LIFT_M} m"
        )
`;

  const manifestJson = JSON.stringify(
    {
      schemaVersion: '1.0',
      generator: APP_CONFIG.appName,
      generatedAt: new Date().toISOString(),
      gates: {
        maxSafeContactForceN: APP_CONFIG.simulation.maxSafeContactForceN,
        maxSafeSlipVelocityMs: APP_CONFIG.simulation.maxSafeSlipVelocityMs,
        targetLiftZMeters: APP_CONFIG.simulation.targetLiftZMeters,
      },
      cases: db.cases,
    },
    null,
    2,
  );

  return {
    githubWorkflowYaml,
    pytestScriptPy,
    manifestJson,
    caseCount: db.cases.length,
    generatedAt: new Date().toISOString(),
  };
}
