/**
 * Command Center Session Summary & Export Bar (`SessionExportBar`)
 *
 * Sits cleanly above the interactive 3D robot stage inside the Command Center
 * stage column (keeping the 3-zone Top Bar untouched). Displays unboxed session
 * metadata (`session.id · robot · status`) alongside a prominent `Export`
 * button with a crisp format menu supporting:
 * - Full Session Report (`.json`) — always available
 * - 20-Env Swarm Telemetry (`.csv`) — available once rollouts execute
 * - Verified Python Controller (`.py`) — available once compiled
 * - Calibrated MuJoCo Scene (`.mjcf.xml`) — available once compiled
 *
 * Use cases:
 * - One-click exporting of telemetry, code, and MuJoCo XML directly from the
 *   main Command Center screen during live demos and regression audits.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  ChevronDown,
  Database,
  Download,
  FileCode2,
  FileJson,
  FileSpreadsheet,
  FileText,
  GitBranch,
  Layers,
} from 'lucide-react';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type { CompilerSessionState } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import {
  exportCompiledMjcfXml,
  exportLatestPolicyPython,
  exportSessionJson,
  exportTelemetryCsv,
} from '../utils/sessionExporter';

interface SessionExportBarProps {
  session: CompilerSessionState;
  robotModel: RobotModelId;
  onExported: (message: string) => void;
  onPromoteToRegressionCi?: () => Promise<void>;
  isPromotingRegression?: boolean;
  onOpenTrainingStudio?: (tab: 'gym_mjx' | 'lerobot') => void;
}

/**
 * Renders the Command Center workspace bar with an interactive Export menu.
 */
export const SessionExportBar: React.FC<SessionExportBarProps> = ({
  session,
  robotModel,
  onExported,
  onPromoteToRegressionCi,
  isPromotingRegression = false,
  onOpenTrainingStudio,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuContainerRef = useRef<HTMLDivElement | null>(null);

  // Close dropdown on outside click or Escape key
  useEffect(() => {
    if (!menuOpen) return;
    const handlePointerDown = (e: MouseEvent) => {
      if (
        menuContainerRef.current &&
        !menuContainerRef.current.contains(e.target as Node)
      ) {
        setMenuOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    window.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [menuOpen]);

  const hasIterations = session.iterations.length > 0;
  const hasCompiledScene = Boolean(session.sceneSpec || session.bundle);
  const latestIter = session.iterations[session.iterations.length - 1];

  /**
   * Executes a selected export action and notifies the user via toast.
   */
  const runExportAction = (format: 'json' | 'csv' | 'py' | 'xml'): void => {
    logFunctionCall('SessionExportBar.runExportAction', {
      sessionId: session.id,
      format,
    });
    setMenuOpen(false);
    if (format === 'json') {
      exportSessionJson(session);
      onExported(`Exported ${session.id}_lettherebesims_report.json`);
      return;
    }
    if (format === 'csv') {
      const ok = exportTelemetryCsv(session);
      onExported(
        ok
          ? `Exported ${session.id}_swarm_telemetry.csv`
          : 'Run the pipeline first to generate 20-environment CSV telemetry.',
      );
      return;
    }
    if (format === 'py') {
      const ok = exportLatestPolicyPython(session);
      onExported(
        ok
          ? `Exported ${session.id}_verified_policy.py`
          : 'Run the pipeline first to compile a Python control policy.',
      );
      return;
    }
    const ok = exportCompiledMjcfXml(session);
    onExported(
      ok
        ? `Exported ${session.id}_scene.mjcf.xml`
        : 'Run the pipeline first to compile a MuJoCo MJCF scene.',
    );
  };

  return (
    <div className="bg-[#111622] border border-[#1E293B] rounded-lg px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
      {/* Unboxed Session Metadata Strip */}
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-400">
        <span className="font-display font-semibold text-slate-100">
          Simulation Stage
        </span>
        <span aria-hidden="true">·</span>
        <span className="font-mono-tabular text-slate-300">
          {APP_CONFIG.robots[robotModel].name}
        </span>
        <span aria-hidden="true">·</span>
        <span className="font-mono-tabular text-slate-400">
          {session.secondaryVideo ? 'Dual View (2 Videos)' : 'Single View'}
        </span>
        {session.sceneSpec && (
          <>
            <span aria-hidden="true">·</span>
            <span
              className={`font-mono-tabular font-medium ${
                session.sceneSpec.executionOutcome === 'success'
                  ? 'text-emerald-400'
                  : session.sceneSpec.executionOutcome === 'hand_demo'
                    ? 'text-amber-300'
                    : session.sceneSpec.executionOutcome === 'no_robot'
                      ? 'text-cyan-400'
                      : 'text-rose-400'
              }`}
            >
              {session.sceneSpec.executionOutcome === 'success'
                ? '✓ Successful Demo'
                : session.sceneSpec.executionOutcome === 'hand_demo'
                  ? '▸ Hand Demo → SO-ARM101'
                  : session.sceneSpec.executionOutcome === 'no_robot'
                    ? '◎ Scene Map'
                    : '✕ Failure Detected'}
            </span>
          </>
        )}
        {session.mobileSensors && (
          <>
            <span aria-hidden="true">·</span>
            <span className="font-mono-tabular text-cyan-300">
              IMU{session.mobileSensors.tapTarget ? ' + Tap' : ''}
              {session.mobileSensors.hasAudioTrack ? ' + Audio' : ''}
            </span>
          </>
        )}
        <span aria-hidden="true">·</span>
        <span className="font-mono-tabular">
          {latestIter
            ? `Pass Rate: ${latestIter.passed}/${latestIter.total}`
            : session.video
              ? session.secondaryVideo
                ? '2 Videos Staged'
                : '1 Video Staged'
              : 'Awaiting Footage'}
        </span>
      </div>

      {/* Prominent Regression CI Promote & Export Controls */}
      <div ref={menuContainerRef} className="relative flex items-center gap-2 shrink-0">
        {onPromoteToRegressionCi && (
          <button
            type="button"
            disabled={!hasCompiledScene || isPromotingRegression}
            onClick={() => void onPromoteToRegressionCi()}
            title="Save this calibrated failure scene into your permanent Regression CI suite"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-cyan-200 bg-[#161D2C] border border-cyan-500/40 rounded-md hover:border-cyan-400 hover:text-cyan-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap cursor-pointer"
          >
            <GitBranch className="w-3.5 h-3.5" />
            <span>{isPromotingRegression ? 'Saving CI...' : 'Save to CI'}</span>
          </button>
        )}

        <button
          type="button"
          onClick={() => runExportAction('json')}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-slate-100 bg-[#161D2C] border border-[#2D3B52] rounded-md hover:border-amber-500/70 hover:text-amber-300 transition-colors whitespace-nowrap cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export</span>
        </button>

        <button
          type="button"
          aria-label="Choose export format"
          aria-expanded={menuOpen}
          onClick={() => {
            logFunctionCall('SessionExportBar.toggleMenu', { nextOpen: !menuOpen });
            setMenuOpen((o) => !o);
          }}
          className="inline-flex items-center justify-center w-8 h-8 text-slate-300 bg-[#161D2C] border border-[#2D3B52] rounded-md hover:text-white hover:border-slate-500 transition-colors cursor-pointer"
        >
          <ChevronDown className="w-3.5 h-3.5" />
        </button>

        {menuOpen && (
          <div
            role="menu"
            className="absolute right-0 top-full mt-1.5 z-40 w-64 bg-[#111622] border border-[#2D3B52] rounded-md py-1 shadow-xl"
          >
            <button
              type="button"
              role="menuitem"
              onClick={() => runExportAction('json')}
              className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] flex items-center justify-between gap-2 cursor-pointer"
            >
              <span className="inline-flex items-center gap-2">
                <FileJson className="w-3.5 h-3.5 text-amber-400" />
                <span>Full Session Report</span>
              </span>
              <span className="font-mono-tabular text-[11px] text-slate-400">.json</span>
            </button>

            <button
              type="button"
              role="menuitem"
              disabled={!hasIterations}
              onClick={() => runExportAction('csv')}
              className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] disabled:opacity-45 disabled:cursor-not-allowed flex items-center justify-between gap-2 cursor-pointer"
            >
              <span className="inline-flex items-center gap-2">
                <FileSpreadsheet className="w-3.5 h-3.5 text-cyan-400" />
                <span>20-Env Swarm Telemetry</span>
              </span>
              <span className="font-mono-tabular text-[11px] text-slate-400">.csv</span>
            </button>

            <button
              type="button"
              role="menuitem"
              disabled={!hasCompiledScene}
              onClick={() => runExportAction('py')}
              className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] disabled:opacity-45 disabled:cursor-not-allowed flex items-center justify-between gap-2 cursor-pointer"
            >
              <span className="inline-flex items-center gap-2">
                <FileCode2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>Verified Controller</span>
              </span>
              <span className="font-mono-tabular text-[11px] text-slate-400">.py</span>
            </button>

            <button
              type="button"
              role="menuitem"
              disabled={!hasCompiledScene}
              onClick={() => runExportAction('xml')}
              className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] disabled:opacity-45 disabled:cursor-not-allowed flex items-center justify-between gap-2 cursor-pointer"
            >
              <span className="inline-flex items-center gap-2">
                <FileText className="w-3.5 h-3.5 text-slate-300" />
                <span>Calibrated MuJoCo Scene</span>
              </span>
              <span className="font-mono-tabular text-[11px] text-slate-400">.xml</span>
            </button>

            {onOpenTrainingStudio && (
              <>
                <div className="my-1 border-t border-[#1E293B]" />
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onOpenTrainingStudio('gym_mjx');
                  }}
                  className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] flex items-center justify-between gap-2 cursor-pointer"
                >
                  <span className="inline-flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5 text-amber-400" />
                    <span>Gymnasium + MJX RL</span>
                  </span>
                  <span className="font-mono-tabular text-[11px] text-amber-300">.zip</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setMenuOpen(false);
                    onOpenTrainingStudio('lerobot');
                  }}
                  className="w-full px-3 py-2 text-left text-xs text-slate-200 hover:bg-[#161D2C] flex items-center justify-between gap-2 cursor-pointer"
                >
                  <span className="inline-flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-cyan-400" />
                    <span>LeRobot v2.1 Dataset</span>
                  </span>
                  <span className="font-mono-tabular text-[11px] text-cyan-300">.zip</span>
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
