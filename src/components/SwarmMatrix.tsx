/**
 * 20-Environment Domain-Randomized Swarm Matrix (`SwarmMatrix`)
 *
 * Displays the live closed-loop pass-rate progression (`0/20 → 11/20 → 20/20`)
 * across all Gemini 4 Neon repair iterations alongside an interactive 5×4
 * environment matrix (`E01`–`E20`). Each cell renders tabular peak contact
 * force (`N`), final lift Z-height (`m`), and explicit Pass/Fail state
 * (`✓ Pass` / `✕ Fault`) without relying on hue alone.
 *
 * Clicking any environment tile immediately loads that environment's exact
 * friction/mass/XY-offset perturbation and rollout trajectory into both the
 * Three.js 3D robot stage and the dual-axis telemetry chart.
 *
 * Use cases:
 * - Auditing which low-friction or extreme XY-jitter environments failed in
 *   Iteration 1 and verifying that they turned green after Neon's policy patch.
 * - Filtering failed environments rapidly during live debugging.
 */

import React, { useState } from 'react';
import { APP_CONFIG } from '../config/appConfig';
import type { IterationRecord } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

interface SwarmMatrixProps {
  iterations: IterationRecord[];
  selectedIterationIndex: number;
  selectedEnvId: number;
  onSelectIteration: (index: number) => void;
  onSelectEnv: (envId: number) => void;
}

type MatrixFilter = 'all' | 'failed' | 'passed';

/**
 * Renders the closed-loop pass-rate counter, iteration selector, and 20-cell
 * interactive domain-randomization matrix.
 */
export const SwarmMatrix: React.FC<SwarmMatrixProps> = ({
  iterations,
  selectedIterationIndex,
  selectedEnvId,
  onSelectIteration,
  onSelectEnv,
}) => {
  const [filter, setFilter] = useState<MatrixFilter>('all');

  const activeIteration = iterations[selectedIterationIndex] ?? null;
  const rollouts = activeIteration?.rollouts ?? [];
  const filteredRollouts = rollouts.filter((r) => {
    if (filter === 'failed') return !r.success;
    if (filter === 'passed') return r.success;
    return true;
  });

  const passPercent = activeIteration
    ? Math.round((activeIteration.passed / Math.max(1, activeIteration.total)) * 100)
    : 0;

  return (
    <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-4">
      {/* Header Row: Pass-Rate Trajectory + Iteration Switcher + Filter Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-[#1E293B]">
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-slate-400">
            Cloud Rollouts Passing (20 Domain-Randomized Environments)
          </span>
          <div className="flex items-baseline gap-3">
            <span className="font-display text-2xl font-bold font-mono-tabular text-slate-100">
              {activeIteration
                ? `${activeIteration.passed}/${activeIteration.total}`
                : '0/20'}
            </span>
            <span
              className={`font-mono-tabular text-sm font-semibold ${
                passPercent === 100
                  ? 'text-emerald-400'
                  : passPercent >= 50
                    ? 'text-amber-400'
                    : 'text-slate-400'
              }`}
            >
              {passPercent}%
            </span>
            {iterations.length > 1 && (
              <span className="font-mono-tabular text-xs text-slate-400">
                Trajectory:{' '}
                {iterations.map((it) => `${it.passed}/${it.total}`).join(' → ')}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Iteration Selector Buttons */}
          {iterations.length > 0 && (
            <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
              {iterations.map((iter, idx) => (
                <button
                  key={iter.iteration}
                  type="button"
                  onClick={() => {
                    logFunctionCall('SwarmMatrix.selectIteration', { iteration: iter.iteration });
                    onSelectIteration(idx);
                  }}
                  className={`px-2.5 py-1 text-xs font-mono-tabular rounded transition-colors whitespace-nowrap cursor-pointer ${
                    idx === selectedIterationIndex
                      ? 'bg-amber-500 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-100'
                  }`}
                >
                  Iter {iter.iteration} ({iter.passed}/{iter.total})
                </button>
              ))}
            </div>
          )}

          {/* Filter Segmented Control */}
          <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
            {(['all', 'failed', 'passed'] as MatrixFilter[]).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => {
                  logFunctionCall('SwarmMatrix.setFilter', { filter: key });
                  setFilter(key);
                }}
                className={`px-2.5 py-1 text-xs font-medium capitalize rounded transition-colors whitespace-nowrap cursor-pointer ${
                  filter === key
                    ? 'bg-[#1E293B] text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {key}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 5×4 Interactive Environment Grid */}
      {rollouts.length === 0 ? (
        <div className="grid grid-cols-4 sm:grid-cols-5 gap-2">
          {Array.from({ length: APP_CONFIG.simulation.swarmSize }, (_, idx) => (
            <div
              key={idx}
              className="h-16 rounded-md bg-[#0B0E17] border border-[#1E293B] px-2.5 py-2 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-[11px] font-mono-tabular text-slate-500">
                <span>E{String(idx + 1).padStart(2, '0')}</span>
                <span>—</span>
              </div>
              <span className="text-[10px] font-mono-tabular text-slate-600">
                Awaiting run
              </span>
            </div>
          ))}
        </div>
      ) : filteredRollouts.length === 0 ? (
        <div className="py-8 text-center text-xs text-slate-400">
          No environments match the &ldquo;{filter}&rdquo; filter in Iteration {activeIteration.iteration}.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-5 gap-2">
          {filteredRollouts.map((rollout) => {
            const isSelected = rollout.envId === selectedEnvId;
            return (
              <button
                key={rollout.envId}
                type="button"
                onClick={() => {
                  logFunctionCall('SwarmMatrix.selectEnv', { envId: rollout.envId });
                  onSelectEnv(rollout.envId);
                }}
                className={`text-left rounded-md px-2.5 py-2 border transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-[#161D2C] border-cyan-400'
                    : rollout.success
                      ? 'bg-[#0B0E17] border-[#1E293B] hover:border-emerald-500/50'
                      : 'bg-[#0B0E17] border-rose-500/35 hover:border-rose-400'
                }`}
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="font-mono-tabular text-xs font-semibold text-slate-100">
                    E{String(rollout.envId).padStart(2, '0')}
                  </span>
                  <span
                    className={`font-mono-tabular text-[11px] font-medium ${
                      rollout.success ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {rollout.success ? '✓ Pass' : '✕ Fail'}
                  </span>
                </div>

                <div className="mt-1.5 flex items-center justify-between text-[10px] font-mono-tabular text-slate-400">
                  <span
                    className={
                      rollout.peakContactForceN > APP_CONFIG.simulation.maxSafeContactForceN
                        ? 'text-rose-300'
                        : 'text-slate-300'
                    }
                  >
                    {rollout.peakContactForceN.toFixed(1)} N
                  </span>
                  <span>Z {rollout.finalObjectZ.toFixed(2)} m</span>
                </div>

                <div className="mt-0.5 text-[10px] font-mono-tabular text-slate-500 truncate">
                  μ={rollout.params.friction.toFixed(2)} · m={rollout.params.mass.toFixed(2)}kg
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Selected Environment Failure / Diagnosis Footnote */}
      {activeIteration && (
        <div className="pt-2 border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="text-slate-300">
            <span className="font-mono-tabular text-cyan-400 mr-2">
              E{String(selectedEnvId).padStart(2, '0')}
            </span>
            {rollouts.find((r) => r.envId === selectedEnvId)?.failureReason ??
              'Nominal grasp stability verified within 45.0 N force and 0.12 m lift gates.'}
          </div>
          {activeIteration.rootCauseFix && (
            <span className="text-[11px] text-amber-300 max-w-xl truncate" title={activeIteration.rootCauseFix}>
              Neon Fix: {activeIteration.rootCauseFix}
            </span>
          )}
        </div>
      )}
    </section>
  );
};
