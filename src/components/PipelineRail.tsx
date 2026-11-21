/**
 * Live 4-Stage Gemini 4 Interactions Pipeline Rail (`PipelineRail`)
 *
 * Displays real-time multi-model orchestration progress across:
 * 1. Gemini 4 Carbon (`thinking_level="high"`) — Spatial Real-to-Sim
 * 2. Cloud Swarm + Gemini 4 Hydrogen (`thinking_level="low"`) — Telemetry Triage
 * 3. Gemini 4 Neon (`thinking_level="xhigh"`) — Adversarial Policy Repair
 * 4. Gemini 4 Neon (`background=true`) — Long Decode Deployment Bundle
 *
 * Renders live `previous_interaction_id` state-chaining metadata as clean
 * unboxed typographic separators (`·`), explicit text/glyph state indicators
 * (zero decorative pulsing dots), and expandable proof-of-work reasoning
 * summaries returned from the Gemini Interactions API.
 *
 * Use cases:
 * - Tracking which Gemini 4 model and thinking depth is active in real time.
 * - Inspecting chain-of-thought summaries and interaction IDs for auditability.
 */

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Link2 } from 'lucide-react';
import type {
  InteractionTraceEntry,
  PipelineProgressState,
  PipelineStageState,
  SessionStatus,
  StageStatus,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { usePipelineProgress } from '../utils/pipelineProgress';

interface PipelineRailProps {
  sessionId: string;
  sessionStatus: SessionStatus;
  progress: PipelineProgressState | null;
  stages: PipelineStageState[];
  interactionTrace: InteractionTraceEntry[];
  webhookEnabled: boolean;
  error: string | null;
}

/**
 * Formats an explicit text-first semantic status indicator without hue-only
 * signaling or pulsing dot animations.
 */
function renderStageStatusLabel(status: StageStatus): { glyph: string; label: string; toneClass: string } {
  switch (status) {
    case 'completed':
      return { glyph: '✓', label: 'Verified', toneClass: 'text-emerald-400' };
    case 'active':
      return { glyph: '▸', label: 'Running', toneClass: 'text-amber-400' };
    case 'skipped':
      return { glyph: '—', label: 'Skipped', toneClass: 'text-slate-400' };
    case 'error':
      return { glyph: '✕', label: 'Fault', toneClass: 'text-rose-400' };
    default:
      return { glyph: '○', label: 'Standby', toneClass: 'text-slate-500' };
  }
}

/**
 * Truncates long interaction IDs cleanly while preserving recognizable suffixes.
 */
function shortInteractionId(id: string | null): string {
  if (!id) return '—';
  return id.length > 16 ? `${id.slice(0, 8)}…${id.slice(-5)}` : id;
}

/**
 * Renders the 4-stage Gemini 4 pipeline tracker and expandable interaction log.
 */
export const PipelineRail: React.FC<PipelineRailProps> = ({
  sessionId,
  sessionStatus,
  progress,
  stages,
  interactionTrace,
  webhookEnabled,
  error,
}) => {
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);
  const displayProgress = usePipelineProgress({
    id: sessionId,
    status: sessionStatus,
    progress,
    stages,
  });

  return (
    <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="font-display text-sm font-semibold text-slate-100">
            02. Gemini 4 Pipeline Rail
          </h2>
          <span className="font-mono-tabular text-[11px] text-slate-400">
            {displayProgress.isRunning || displayProgress.isCompleted
              ? `${displayProgress.percent}% · ${displayProgress.elapsedLabel}`
              : webhookEnabled
                ? 'Cloud Webhook'
                : 'Built-In Physics'}
          </span>
        </div>

        {(displayProgress.isRunning || displayProgress.isCompleted || displayProgress.isError) && (
          <div className="w-full h-1 bg-[#161D2C] rounded overflow-hidden">
            <div
              className={`h-full transition-[width] duration-300 ease-out ${
                displayProgress.isCompleted
                  ? 'bg-emerald-500'
                  : displayProgress.isError
                    ? 'bg-rose-500'
                    : 'bg-amber-500'
              }`}
              style={{ width: `${displayProgress.percent}%` }}
            />
          </div>
        )}
      </div>

      {error && (
        <div className="p-3 rounded-md bg-rose-500/10 border border-rose-500/35 text-xs text-rose-200 leading-relaxed">
          <p className="font-semibold text-rose-300">Pipeline Halted</p>
          <p className="mt-0.5">{error}</p>
        </div>
      )}

      {/* 4 Stateful Stages */}
      <ol className="flex flex-col divide-y divide-[#1E293B]">
        {stages.map((stage) => {
          const badge = renderStageStatusLabel(stage.status);
          return (
            <li key={stage.key} className="py-3 first:pt-0 last:pb-0 flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-100">
                  {stage.label}
                </span>
                <span className={`font-mono-tabular text-[11px] font-medium shrink-0 ${badge.toneClass}`}>
                  {badge.glyph} {badge.label}
                </span>
              </div>

              {/* Clean unboxed metadata separated by typographic dots */}
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] font-mono-tabular text-slate-400">
                <span className="text-slate-300">{stage.model}</span>
                <span aria-hidden="true">·</span>
                <span>thinking={stage.thinkingLevel}</span>
                {stage.interactionId && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="text-cyan-400" title={stage.interactionId}>
                      id:{shortInteractionId(stage.interactionId)}
                    </span>
                  </>
                )}
              </div>

              {stage.previousInteractionId && (
                <div className="flex items-center gap-1 text-[10px] font-mono-tabular text-slate-500">
                  <Link2 className="w-3 h-3 text-slate-500 shrink-0" />
                  <span title={stage.previousInteractionId}>
                    prev_interaction_id: {shortInteractionId(stage.previousInteractionId)}
                  </span>
                </div>
              )}

              <p className="text-[11px] leading-relaxed text-slate-400">
                {stage.detail}
              </p>
            </li>
          );
        })}
      </ol>

      {/* Expandable Proof-of-Work Interaction Trace */}
      {interactionTrace.length > 0 && (
        <div className="pt-3 border-t border-[#1E293B] flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-200">
              Interaction Chain Trace
            </h3>
            <span className="font-mono-tabular text-[11px] text-slate-400">
              {interactionTrace.length} turn{interactionTrace.length === 1 ? '' : 's'}
            </span>
          </div>

          <div className="flex flex-col gap-1.5 max-h-64 overflow-y-auto pr-1">
            {interactionTrace.map((entry) => {
              const isOpen = expandedTraceId === entry.id;
              return (
                <div
                  key={entry.id}
                  className="bg-[#0B0E17] border border-[#1E293B] rounded-md"
                >
                  <button
                    type="button"
                    onClick={() => {
                      logFunctionCall('PipelineRail.toggleTrace', { traceId: entry.id });
                      setExpandedTraceId(isOpen ? null : entry.id);
                    }}
                    className="w-full px-2.5 py-2 flex items-center justify-between gap-2 text-left cursor-pointer hover:bg-[#141B29] transition-colors"
                  >
                    <div className="min-w-0">
                      <p className="text-[11px] font-medium text-slate-200 truncate">
                        {entry.title}
                      </p>
                      <p className="font-mono-tabular text-[10px] text-slate-400">
                        {entry.model} · {entry.thinkingLevel} · {(entry.durationMs / 1000).toFixed(1)}s
                      </p>
                    </div>
                    {isOpen ? (
                      <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    )}
                  </button>

                  {isOpen && (
                    <div className="px-2.5 pb-2.5 pt-1 border-t border-[#1E293B] flex flex-col gap-2 text-[11px]">
                      {entry.thoughtSummary && (
                        <div>
                          <span className="font-mono-tabular text-[10px] text-amber-400">
                            Reasoning Summary
                          </span>
                          <p className="mt-0.5 text-slate-300 leading-relaxed whitespace-pre-wrap">
                            {entry.thoughtSummary}
                          </p>
                        </div>
                      )}
                      <div>
                        <span className="font-mono-tabular text-[10px] text-cyan-400">
                          Turn Output
                        </span>
                        <p className="mt-0.5 text-slate-400 leading-relaxed whitespace-pre-wrap">
                          {entry.outputExcerpt}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
};
