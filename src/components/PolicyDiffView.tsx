/**
 * Policy Diff & Compiled MJCF XML Inspector (`PolicyDiffView`)
 *
 * Allows robotics engineers to audit every exact edit made by Gemini 4 Neon
 * (`thinking_level="xhigh"`) to `execute_policy(model, data, step)` across
 * closed-loop repair iterations, compare structured impedance/approach gain
 * tables, and inspect or export the calibrated MuJoCo `MJCF` XML compiled by
 * Gemini 4 Carbon.
 *
 * Use cases:
 * - Reviewing how Neon fixed contact-force spikes or grasp slip between Iter 1
 *   and Iter 3 via a clean line-by-line diff.
 * - Copying or downloading the compiled `.xml` MuJoCo scene or `.py` controller.
 */

import React, { useMemo, useState } from 'react';
import { Check, Copy, Download } from 'lucide-react';
import type { IterationRecord, SceneSpec } from '../types/robotics';
import { computePolicyDiff } from '../utils/diffEngine';
import { copyTextToClipboard, downloadTextArtifact } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

interface PolicyDiffViewProps {
  sceneSpec: SceneSpec | null;
  iterations: IterationRecord[];
  selectedIterationIndex: number;
  onSelectIteration: (idx: number) => void;
}

/**
 * Renders the Python policy diff viewer, controller parameter comparison grid,
 * and MuJoCo MJCF XML inspector.
 */
export const PolicyDiffView: React.FC<PolicyDiffViewProps> = ({
  sceneSpec,
  iterations,
  selectedIterationIndex,
  onSelectIteration,
}) => {
  const [activeTab, setActiveTab] = useState<'policy_diff' | 'mjcf_xml'>('policy_diff');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const activeIteration = iterations[selectedIterationIndex] ?? null;
  const activePolicyCode =
    activeIteration?.policyCode ?? sceneSpec?.initialControllerPython ?? '';

  const diffResult = useMemo(
    () =>
      computePolicyDiff(
        activeIteration?.previousPolicyCode ?? null,
        activePolicyCode,
      ),
    [activeIteration?.previousPolicyCode, activePolicyCode],
  );

  /**
   * Copies the active code artifact to clipboard with visual confirmation.
   */
  const handleCopy = async (key: string, text: string): Promise<void> => {
    logFunctionCall('PolicyDiffView.handleCopy', { key, length: text.length });
    const ok = await copyTextToClipboard(text);
    if (ok) {
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey(null), 1800);
    }
  };

  if (!sceneSpec) {
    return (
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-10 text-center">
        <h2 className="font-display text-lg font-semibold text-slate-100">
          No Compiled Policy or MJCF Scene Yet
        </h2>
        <p className="mt-2 text-sm text-slate-400 max-w-lg mx-auto">
          Upload a smartphone or desktop tabletop video in the Command Center and run the Gemini 4 pipeline to inspect line-by-line policy patches and the calibrated MuJoCo XML.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Top Toolbar */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
          <button
            type="button"
            onClick={() => {
              logFunctionCall('PolicyDiffView.selectTab', { tab: 'policy_diff' });
              setActiveTab('policy_diff');
            }}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer ${
              activeTab === 'policy_diff'
                ? 'bg-[#1E293B] text-white'
                : 'text-slate-400 hover:text-slate-100'
            }`}
          >
            Python Policy Diff
          </button>
          <button
            type="button"
            onClick={() => {
              logFunctionCall('PolicyDiffView.selectTab', { tab: 'mjcf_xml' });
              setActiveTab('mjcf_xml');
            }}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer ${
              activeTab === 'mjcf_xml'
                ? 'bg-[#1E293B] text-white'
                : 'text-slate-400 hover:text-slate-100'
            }`}
          >
            Compiled MuJoCo MJCF XML
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {activeTab === 'policy_diff' && iterations.length > 0 && (
            <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
              {iterations.map((iter, idx) => (
                <button
                  key={iter.iteration}
                  type="button"
                  onClick={() => onSelectIteration(idx)}
                  className={`px-2.5 py-1 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                    idx === selectedIterationIndex
                      ? 'bg-amber-500 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Iter {iter.iteration} ({iter.passed}/{iter.total})
                </button>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={() =>
              void handleCopy(
                activeTab,
                activeTab === 'policy_diff' ? activePolicyCode : sceneSpec.mjcfXml,
              )
            }
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded hover:border-[#2D3B52] transition-colors cursor-pointer"
          >
            {copiedKey === activeTab ? (
              <Check className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
            <span>{copiedKey === activeTab ? 'Copied' : 'Copy'}</span>
          </button>

          <button
            type="button"
            onClick={() =>
              downloadTextArtifact(
                activeTab === 'policy_diff'
                  ? `verified_policy_iter_${activeIteration?.iteration ?? 1}.py`
                  : 'calibrated_scene.mjcf.xml',
                activeTab === 'policy_diff' ? activePolicyCode : sceneSpec.mjcfXml,
              )
            }
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-950 bg-amber-500 rounded hover:bg-amber-400 transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download</span>
          </button>
        </div>
      </div>

      {activeTab === 'policy_diff' ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left Diagnostics & Gain Parameters */}
          <div className="lg:col-span-4 flex flex-col gap-4">
            <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  {sceneSpec.executionOutcome === 'success'
                    ? 'Demonstration Robustness Analysis'
                    : sceneSpec.executionOutcome === 'no_robot'
                      ? 'Spatial Scene Calibration'
                      : 'Root-Cause Failure Diagnosis'}
                </h3>
                <span
                  className={`font-mono-tabular text-[11px] font-medium ${
                    sceneSpec.executionOutcome === 'success'
                      ? 'text-emerald-400'
                      : sceneSpec.executionOutcome === 'no_robot'
                        ? 'text-cyan-400'
                        : 'text-rose-400'
                  }`}
                >
                  {sceneSpec.executionOutcome === 'success'
                    ? '✓ Success'
                    : sceneSpec.executionOutcome === 'no_robot'
                      ? '◎ Scene Map'
                      : '✕ Failure'}
                </span>
              </div>
              {sceneSpec.outcomeSummary && (
                <p className="text-xs font-medium text-slate-200 leading-relaxed">
                  {sceneSpec.outcomeSummary}
                </p>
              )}
              <p className="text-xs leading-relaxed text-slate-300">
                {activeIteration?.rootCauseFix ||
                  sceneSpec.failureAnalysis}
              </p>
              {activeIteration?.telemetrySummary && (
                <div className="mt-2 pt-3 border-t border-[#1E293B]">
                  <span className="font-mono-tabular text-[11px] text-cyan-400">
                    Hydrogen Telemetry Triage
                  </span>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400 whitespace-pre-wrap max-h-48 overflow-y-auto">
                    {activeIteration.telemetrySummary}
                  </p>
                </div>
              )}
            </div>

            {activeIteration && (
              <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Active Control Parameters
                </h3>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-xs">
                  <div>
                    <dt className="text-slate-400">Approach Height</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.approachHeightM.toFixed(3)} m
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Clearance Arc</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.clearanceArcM.toFixed(3)} m
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">PD Stiffness Kp</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.kpStiffness.toFixed(1)} N/m
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">PD Damping Kd</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.kdDamping.toFixed(1)} Ns/m
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Grip Force</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.gripForceN.toFixed(1)} N
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Grip Ramp</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.gripRampSteps} steps
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Descent Speed</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.descentSpeedMs.toFixed(3)} m/s
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Lift Target</dt>
                    <dd className="font-mono-tabular text-slate-100 mt-0.5">
                      {activeIteration.controllerParams.liftHeightM.toFixed(3)} m
                    </dd>
                  </div>
                </dl>
              </div>
            )}
          </div>

          {/* Right Line-by-Line Python Diff */}
          <div className="lg:col-span-8 bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden flex flex-col">
            <div className="px-4 py-2.5 border-b border-[#1E293B] flex items-center justify-between text-xs font-mono-tabular">
              <span className="text-slate-200">
                execute_policy(model, data, step) · Iter {activeIteration?.iteration ?? 1}
              </span>
              <div className="flex items-center gap-3">
                <span className="text-emerald-400">+{diffResult.addedCount}</span>
                <span className="text-rose-400">-{diffResult.removedCount}</span>
              </div>
            </div>
            <div className="overflow-auto max-h-[540px] bg-[#0B0E17] font-mono text-xs leading-6">
              {diffResult.lines.map((line, i) => (
                <div
                  key={i}
                  className={`flex px-3 ${
                    line.type === 'added'
                      ? 'bg-emerald-500/12 text-emerald-200'
                      : line.type === 'removed'
                        ? 'bg-rose-500/12 text-rose-200'
                        : 'text-slate-300'
                  }`}
                >
                  <span className="w-9 shrink-0 select-none text-right pr-3 text-slate-600 font-mono-tabular">
                    {line.oldLineNumber ?? ''}
                  </span>
                  <span className="w-9 shrink-0 select-none text-right pr-3 text-slate-600 font-mono-tabular">
                    {line.newLineNumber ?? ''}
                  </span>
                  <span className="w-5 shrink-0 select-none">
                    {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
                  </span>
                  <pre className="whitespace-pre-wrap break-all flex-1">{line.text}</pre>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Compiled MuJoCo MJCF XML Viewer */
        <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 border-b border-[#1E293B] flex items-center justify-between text-xs font-mono-tabular">
            <span className="text-slate-200">
              calibrated_scene.mjcf.xml · Target: {sceneSpec.targetObject.name}
            </span>
            <span className="text-slate-400">
              {sceneSpec.obstacles.length} obstacle(s) · 20 randomized variants
            </span>
          </div>
          <pre className="p-4 overflow-auto max-h-[560px] bg-[#0B0E17] font-mono text-xs leading-6 text-cyan-100 whitespace-pre">
            {sceneSpec.mjcfXml}
          </pre>
        </div>
      )}
    </div>
  );
};
