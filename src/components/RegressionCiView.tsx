/**
 * Failure-to-Regression-Suite CI/CD Workbench View (`RegressionCiView.tsx`)
 *
 * Implements "pytest for Physical Robots" inside LetThereBeSims:
 * 1. Promotes real-world failure sessions into permanent, disk-persisted
 *    MuJoCo regression test cases (each covering 20 randomized physics variants).
 * 2. Runs the entire Regression Suite CI gate on demand against:
 *    - The active session's latest compiled policy (`candidate_policy`),
 *    - Every case's stored verified patch (`suite_verified`), or
 *    - Every case's original unpatched baseline (`suite_baseline`, verifying
 *      that CI catches regressions with `Exit 1`).
 * 3. Displays chronological CI build history (`PASS · Exit 0` / `FAIL · Exit 1`)
 *    and exports ready-to-commit GitHub Actions YAML, `pytest` harness, and
 *    JSON regression suite manifest.
 *
 * Use cases:
 * - Ensuring a repaired grasp failure never regresses when updating policies.
 * - Exporting a drop-in GitHub Actions workflow + `pytest` gate for any repo.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Check,
  CheckCircle2,
  Copy,
  Download,
  GitBranch,
  Play,
  Plus,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Terminal,
  Trash2,
  XCircle,
} from 'lucide-react';
import { APP_CONFIG } from '../config/appConfig';
import {
  deleteRegressionTestCase,
  fetchRegressionCiArtifacts,
  fetchRegressionSuite,
  runRegressionSuiteCiApi,
  saveSessionToRegressionSuite,
} from '../services/compilerApi';
import type {
  CompilerSessionState,
  RegressionCiArtifacts,
  RegressionRunMode,
  RegressionSuiteSnapshot,
  RegressionTestCase,
} from '../types/robotics';
import { copyTextToClipboard, downloadTextArtifact } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

interface RegressionCiViewProps {
  session: CompilerSessionState;
  onNotify: (message: string) => void;
}

type CiArtifactTab = 'github_actions' | 'pytest' | 'manifest';

/**
 * Renders the Failure-to-Regression-Suite CI/CD dashboard, live suite runner,
 * saved failure case bank, build history, and GitHub Actions / pytest exporter.
 */
export const RegressionCiView: React.FC<RegressionCiViewProps> = ({
  session,
  onNotify,
}) => {
  const [suite, setSuite] = useState<RegressionSuiteSnapshot>({ cases: [], runs: [] });
  const [artifacts, setArtifacts] = useState<RegressionCiArtifacts | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSavingCase, setIsSavingCase] = useState(false);
  const [isRunningCi, setIsRunningCi] = useState(false);
  const [runMode, setRunMode] = useState<RegressionRunMode>('suite_verified');
  const [customCaseName, setCustomCaseName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [artifactTab, setArtifactTab] = useState<CiArtifactTab>('github_actions');
  const [copiedArtifact, setCopiedArtifact] = useState(false);
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);

  const hasActiveCompiledScene = Boolean(session.sceneSpec);
  const hasActiveCandidatePolicy = Boolean(
    session.iterations.length > 0 || session.sceneSpec?.controllerParams,
  );

  /**
   * Loads saved regression cases, CI build history, and generated CI files.
   */
  const loadSuiteAndArtifacts = useCallback(async (): Promise<void> => {
    logFunctionCall('RegressionCiView.loadSuiteAndArtifacts');
    setIsLoading(true);
    setError(null);
    try {
      const [snapshot, ciFiles] = await Promise.all([
        fetchRegressionSuite(),
        fetchRegressionCiArtifacts(),
      ]);
      setSuite(snapshot);
      setArtifacts(ciFiles);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load Regression CI suite.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSuiteAndArtifacts();
  }, [loadSuiteAndArtifacts]);

  /**
   * Promotes the current session's calibrated failure scene & fix into the suite.
   */
  const handlePromoteActiveSession = async (): Promise<void> => {
    if (!session.sceneSpec) return;
    logFunctionCall('RegressionCiView.handlePromoteActiveSession', {
      sessionId: session.id,
      customCaseName,
    });
    setIsSavingCase(true);
    setError(null);
    try {
      const { testCase, suite: updated } = await saveSessionToRegressionSuite({
        sessionId: session.id,
        name: customCaseName.trim() || undefined,
      });
      setSuite(updated);
      setCustomCaseName('');
      const refreshedArtifacts = await fetchRegressionCiArtifacts();
      setArtifacts(refreshedArtifacts);
      onNotify(`Promoted "${testCase.name}" to Regression CI Suite.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to save regression case.');
    } finally {
      setIsSavingCase(false);
    }
  };

  /**
   * Deletes a saved regression test case and refreshes CI export artifacts.
   */
  const handleDeleteCase = async (testCase: RegressionTestCase): Promise<void> => {
    logFunctionCall('RegressionCiView.handleDeleteCase', { caseId: testCase.id });
    setError(null);
    try {
      const updated = await deleteRegressionTestCase(testCase.id);
      setSuite(updated);
      const refreshedArtifacts = await fetchRegressionCiArtifacts();
      setArtifacts(refreshedArtifacts);
      onNotify(`Removed "${testCase.name}" from Regression CI Suite.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete regression case.');
    }
  };

  /**
   * Executes the full Regression Suite CI gate across every saved failure case.
   */
  const handleRunRegressionCi = async (overrideMode?: RegressionRunMode): Promise<void> => {
    const selectedMode = overrideMode ?? runMode;
    logFunctionCall('RegressionCiView.handleRunRegressionCi', {
      mode: selectedMode,
      sessionId: session.id,
    });
    setIsRunningCi(true);
    setError(null);
    try {
      const policyLabel =
        selectedMode === 'candidate_policy'
          ? `Session ${session.id} (${APP_CONFIG.robots[session.robotModel].name})`
          : selectedMode === 'suite_baseline'
            ? 'Unpatched Baseline Controllers (Regression Check)'
            : 'Verified Patched Controllers';
      const { run, suite: updated } = await runRegressionSuiteCiApi({
        mode: selectedMode,
        sessionId: selectedMode === 'candidate_policy' ? session.id : undefined,
        policyLabel,
      });
      setSuite(updated);
      onNotify(
        run.overallStatus === 'pass'
          ? `CI Gate PASSED (Exit 0): ${run.passedCases}/${run.totalCases} cases · ${run.totalEnvsPassed}/${run.totalEnvsEvaluated} envs.`
          : `CI Gate FAILED (Exit 1): ${run.failedCases}/${run.totalCases} failure case(s) regressed.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Regression CI execution failed.');
    } finally {
      setIsRunningCi(false);
    }
  };

  const latestRun = suite.runs[0] ?? null;
  const totalRandomizedEnvs = suite.cases.length * APP_CONFIG.simulation.swarmSize;

  const activeArtifactContent =
    artifactTab === 'github_actions'
      ? {
          filename: 'robotics-regression.yml',
          mime: 'text/yaml;charset=utf-8',
          pathLabel: '.github/workflows/robotics-regression.yml',
          content: artifacts?.githubWorkflowYaml ?? '',
        }
      : artifactTab === 'pytest'
        ? {
            filename: 'test_robotics_regression.py',
            mime: 'text/x-python;charset=utf-8',
            pathLabel: 'tests/test_robotics_regression.py',
            content: artifacts?.pytestScriptPy ?? '',
          }
        : {
            filename: 'regression_suite_manifest.json',
            mime: 'application/json;charset=utf-8',
            pathLabel: 'tests/regression_suite_manifest.json',
            content: artifacts?.manifestJson ?? '',
          };

  return (
    <div className="flex flex-col gap-5">
      {/* Header & Top CI Gate Summary */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-5 flex flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex flex-col gap-1 max-w-2xl">
            <div className="flex items-center gap-2.5">
              <GitBranch className="w-4 h-4 text-amber-400" />
              <h2 className="font-display text-base font-semibold text-slate-100">
                Failure-to-Regression-Suite CI/CD
              </h2>
              <span className="font-mono-tabular text-[11px] text-cyan-400">
                pytest for Physical Robots
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Every real-world failure clip you verify becomes a permanent MuJoCo regression test case
              (20 domain-randomized environments each). Run the full CI gate before deploying any policy
              update so previously fixed collisions, slips, or incomplete placements can never regress.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={isLoading}
              onClick={() => void loadSuiteAndArtifacts()}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded-md hover:border-slate-500 disabled:opacity-50 transition-colors cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Metrics Strip */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-2 border-t border-[#1E293B]">
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3">
            <p className="text-[11px] text-slate-400">Saved Regression Cases</p>
            <p className="mt-1 font-mono-tabular text-lg font-semibold text-slate-100">
              {suite.cases.length}
            </p>
          </div>
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3">
            <p className="text-[11px] text-slate-400">Randomized CI Matrix</p>
            <p className="mt-1 font-mono-tabular text-lg font-semibold text-cyan-300">
              {totalRandomizedEnvs} envs
            </p>
          </div>
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3">
            <p className="text-[11px] text-slate-400">Latest CI Gate Status</p>
            <p
              className={`mt-1 font-mono-tabular text-lg font-semibold ${
                !latestRun
                  ? 'text-slate-500'
                  : latestRun.overallStatus === 'pass'
                    ? 'text-emerald-400'
                    : 'text-rose-400'
              }`}
            >
              {!latestRun
                ? 'Not Run'
                : latestRun.overallStatus === 'pass'
                  ? 'PASS · Exit 0'
                  : 'FAIL · Exit 1'}
            </p>
          </div>
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3">
            <p className="text-[11px] text-slate-400">Latest Env Pass Rate</p>
            <p className="mt-1 font-mono-tabular text-lg font-semibold text-amber-300">
              {latestRun
                ? `${latestRun.totalEnvsPassed}/${latestRun.totalEnvsEvaluated} (${Math.round(
                    (latestRun.totalEnvsPassed / Math.max(1, latestRun.totalEnvsEvaluated)) * 100,
                  )}%)`
                : '—'}
            </p>
          </div>
        </div>

        {/* Promote Active Session + Execute CI Runner Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 pt-2 border-t border-[#1E293B]">
          {/* Promote Active Session */}
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3.5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-slate-200">
                Promote Active Session ({session.id})
              </span>
              <span className="font-mono-tabular text-[10px] text-slate-400">
                {hasActiveCompiledScene ? 'Scene Ready' : 'Compile a video first'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={customCaseName}
                onChange={(e) => setCustomCaseName(e.target.value)}
                disabled={!hasActiveCompiledScene || isSavingCase}
                placeholder={
                  session.sceneSpec
                    ? `${session.sceneSpec.targetObject.name} · Failure Gate`
                    : 'Optional regression case name...'
                }
                className="flex-1 min-w-0 px-3 py-1.5 text-xs text-slate-100 bg-[#111622] border border-[#1E293B] rounded focus:outline-none focus:border-amber-500 disabled:opacity-50"
              />
              <button
                type="button"
                disabled={!hasActiveCompiledScene || isSavingCase}
                onClick={() => void handlePromoteActiveSession()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-amber-500 rounded hover:bg-amber-400 disabled:bg-[#1E293B] disabled:text-slate-500 transition-colors whitespace-nowrap cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>{isSavingCase ? 'Saving...' : 'Save to Suite'}</span>
              </button>
            </div>
          </div>

          {/* Execute CI Gate */}
          <div className="bg-[#0B0E17] border border-[#1E293B] rounded-md p-3.5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-slate-200">
                Run Regression CI Gate
              </span>
              <span className="font-mono-tabular text-[10px] text-slate-400">
                Gate: ≤45N · ≤0.08 m/s · ≥0.12m · Full Place
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={runMode}
                onChange={(e) => setRunMode(e.target.value as RegressionRunMode)}
                disabled={isRunningCi}
                className="flex-1 min-w-[190px] px-2.5 py-1.5 text-xs text-slate-100 bg-[#111622] border border-[#1E293B] rounded focus:outline-none focus:border-amber-500"
              >
                <option value="suite_verified">Verify Stored Patched Policies</option>
                <option value="candidate_policy" disabled={!hasActiveCandidatePolicy}>
                  Test Active Session Policy ({session.id})
                </option>
                <option value="suite_baseline">Test Unpatched Baselines (Catch Regressions)</option>
              </select>
              <button
                type="button"
                disabled={suite.cases.length === 0 || isRunningCi}
                onClick={() => void handleRunRegressionCi()}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-emerald-400 rounded hover:bg-emerald-300 disabled:bg-[#1E293B] disabled:text-slate-500 transition-colors whitespace-nowrap cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>{isRunningCi ? 'Running CI...' : 'Run Full Suite'}</span>
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-md bg-rose-500/10 border border-rose-500/35 text-xs text-rose-200">
            {error}
          </div>
        )}
      </div>

      {/* Saved Regression Failure Cases Table */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[#1E293B] flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Saved Physical Failure Regression Bank
            </h3>
          </div>
          <span className="font-mono-tabular text-[11px] text-slate-400">
            Auto-promoted on completed repairs · Persisted on server
          </span>
        </div>

        {suite.cases.length === 0 ? (
          <div className="p-8 text-center flex flex-col items-center gap-2">
            <ShieldAlert className="w-7 h-7 text-slate-500" />
            <p className="text-sm font-medium text-slate-200">
              No Regression Test Cases Saved Yet
            </p>
            <p className="text-xs text-slate-400 max-w-md leading-relaxed">
              Upload a robot video in the Command Center and run{' '}
              <span className="text-amber-300 font-medium">Analyze &amp; Verify Policy</span>. Completed
              failure repairs auto-promote here, or click <span className="text-amber-300 font-medium">Save to Suite</span> above.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-[#1E293B]">
            {suite.cases.map((tc) => {
              const run = tc.lastRun;
              const isExpanded = expandedCaseId === tc.id;
              return (
                <div key={tc.id} className="p-4 flex flex-col gap-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="flex flex-col gap-1 min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-mono-tabular text-[11px] text-slate-400">
                          {tc.id}
                        </span>
                        <span className="font-display text-sm font-semibold text-slate-100">
                          {tc.name}
                        </span>
                        <span className="font-mono-tabular text-[11px] text-cyan-300">
                          {APP_CONFIG.robots[tc.robotModel]?.name ?? tc.robotModel}
                        </span>
                        {run && (
                          <span
                            className={`inline-flex items-center gap-1 font-mono-tabular text-[11px] font-semibold ${
                              run.status === 'pass' ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {run.status === 'pass' ? (
                              <CheckCircle2 className="w-3.5 h-3.5" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5" />
                            )}
                            <span>
                              {run.passedEnvs}/{run.totalEnvs} envs ({run.status.toUpperCase()})
                            </span>
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {tc.failureSummary}
                      </p>
                      <div className="flex items-center gap-3 flex-wrap font-mono-tabular text-[10px] text-slate-500">
                        <span>Target: {tc.sceneSpec.targetObject.name}</span>
                        <span>Obstacles: {tc.sceneSpec.obstacles.length}</span>
                        {run && (
                          <>
                            <span>Peak Force: {run.worstContactForceN} N</span>
                            <span>Max Slip: {run.worstSlipVelocityMs} m/s</span>
                            <span>Tasks Completed: {run.tasksCompletedEnvs}/{run.totalEnvs}</span>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() => setExpandedCaseId(isExpanded ? null : tc.id)}
                        className="px-2.5 py-1.5 text-[11px] font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded hover:border-slate-500 cursor-pointer"
                      >
                        {isExpanded ? 'Hide MJCF' : 'Inspect MJCF'}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          downloadTextArtifact(
                            `${tc.id}_regression.xml`,
                            tc.sceneSpec.mjcfXml,
                            'application/xml;charset=utf-8',
                          );
                          onNotify(`Downloaded ${tc.id}_regression.xml`);
                        }}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded hover:border-amber-500/60 cursor-pointer"
                      >
                        <Download className="w-3 h-3" />
                        <span>.xml</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDeleteCase(tc)}
                        title="Delete regression test case"
                        className="inline-flex items-center justify-center w-7 h-7 text-slate-400 bg-[#161D2C] border border-[#2D3B52] rounded hover:text-rose-300 hover:border-rose-500/50 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {run && run.failureReasons.length > 0 && (
                    <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-[11px] text-rose-200 font-mono-tabular">
                      {run.failureReasons.map((reason, idx) => (
                        <div key={idx}>• {reason}</div>
                      ))}
                    </div>
                  )}

                  {isExpanded && (
                    <pre className="p-3 rounded bg-[#07090E] border border-[#1E293B] font-mono-tabular text-[11px] text-slate-300 overflow-x-auto max-h-60">
                      <code>{tc.sceneSpec.mjcfXml}</code>
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* CI Build Run History */}
      {suite.runs.length > 0 && (
        <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-[#1E293B] flex items-center justify-between gap-2">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              CI Gate Run History
            </h3>
            <span className="font-mono-tabular text-[11px] text-slate-400">
              Last {suite.runs.length} build(s)
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#0B0E17] text-[11px] text-slate-400 border-b border-[#1E293B]">
                <tr>
                  <th className="px-4 py-2 font-medium">Build ID</th>
                  <th className="px-4 py-2 font-medium">Policy Tested</th>
                  <th className="px-4 py-2 font-medium">Cases Passed</th>
                  <th className="px-4 py-2 font-medium">Envs Passed</th>
                  <th className="px-4 py-2 font-medium">Duration</th>
                  <th className="px-4 py-2 font-medium">CI Gate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B] font-mono-tabular">
                {suite.runs.map((run) => (
                  <tr key={run.runId} className="hover:bg-[#161D2C]/40">
                    <td className="px-4 py-2.5 text-slate-300">{run.runId}</td>
                    <td className="px-4 py-2.5 font-sans text-slate-200">{run.policyLabel}</td>
                    <td className="px-4 py-2.5 text-slate-300">
                      {run.passedCases}/{run.totalCases}
                    </td>
                    <td className="px-4 py-2.5 text-slate-300">
                      {run.totalEnvsPassed}/{run.totalEnvsEvaluated}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400">{run.durationMs} ms</td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`font-semibold ${
                          run.overallStatus === 'pass' ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {run.overallStatus === 'pass' ? 'PASS (Exit 0)' : 'FAIL (Exit 1)'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Ready-to-Commit GitHub Actions + pytest CI/CD Exporter */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-cyan-400" />
            <h3 className="font-display text-sm font-semibold text-slate-100">
              GitHub Actions &amp; pytest CI/CD Files
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
              {(
                [
                  { id: 'github_actions' as const, label: 'GitHub Actions (.yml)' },
                  { id: 'pytest' as const, label: 'pytest Suite (.py)' },
                  { id: 'manifest' as const, label: 'Suite Manifest (.json)' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setArtifactTab(tab.id)}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors cursor-pointer ${
                    artifactTab === tab.id
                      ? 'bg-[#1E293B] text-amber-300'
                      : 'text-slate-400 hover:text-slate-100'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={async () => {
                const ok = await copyTextToClipboard(activeArtifactContent.content);
                if (ok) {
                  setCopiedArtifact(true);
                  window.setTimeout(() => setCopiedArtifact(false), 1800);
                }
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded hover:border-slate-500 cursor-pointer"
            >
              {copiedArtifact ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span>{copiedArtifact ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              type="button"
              onClick={() => {
                downloadTextArtifact(
                  activeArtifactContent.filename,
                  activeArtifactContent.content,
                  activeArtifactContent.mime,
                );
                onNotify(`Downloaded ${activeArtifactContent.filename}`);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-semibold text-slate-950 bg-amber-500 rounded hover:bg-amber-400 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </button>
          </div>
        </div>

        <div className="px-4 py-2 bg-[#0B0E17] border-b border-[#1E293B] font-mono-tabular text-[11px] text-slate-400">
          {activeArtifactContent.pathLabel}
        </div>
        <pre className="p-4 bg-[#07090E] font-mono-tabular text-xs text-slate-200 overflow-x-auto max-h-[420px] leading-relaxed">
          <code>{activeArtifactContent.content || '// Loading CI artifacts...'}</code>
        </pre>
      </div>
    </div>
  );
};
