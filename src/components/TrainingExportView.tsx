/**
 * Robot Training & Export Studio (`TrainingExportView.tsx`)
 *
 * Unifies all robotics training and hardware deployment exports inside
 * LetThereBeSims under three crisp workbench modes:
 * 1. `Gymnasium + MJX RL`: Configures, previews, and downloads a complete
 *    runnable Reinforcement Learning package (`env.py`, `mjx_env.py`,
 *    `train_rl.py`, RL-ready `scene.mjcf.xml`, `requirements.txt`, `README.md`)
 *    calibrated to the active session or any saved Regression CI scene.
 * 2. `LeRobot Dataset`: Generates N domain-randomized synthetic pick-and-place
 *    episodes via the calibrated physics/IK evaluator, filters to verified
 *    passes, computes per-feature LeRobot v2.1 normalization statistics, and
 *    exports a full `.zip` dataset ready for ACT, Diffusion Policy, or SmolVLA.
 * 3. `Deployment Bundle`: Displays the asynchronous Stage 4 Long Decode
 *    production bundle (`.py`, `.xml`, `.md`).
 *
 * Use cases:
 * - Turning a single calibrated phone video into an immediately trainable
 *   Gymnasium / MuJoCo MJX RL environment or a 25–100 episode LeRobot dataset.
 * - Exporting directly from saved Regression CI cases even across sessions.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Check,
  Copy,
  Database,
  Download,
  FileArchive,
  Layers,
  Package,
  RefreshCw,
} from 'lucide-react';
import { APP_CONFIG } from '../config/appConfig';
import {
  downloadGymMjxPackageZip,
  downloadLeRobotDatasetZip,
  fetchRegressionSuite,
  previewGymMjxPackage,
  previewLeRobotDataset,
} from '../services/compilerApi';
import type {
  CompilerSessionState,
  GymMjxPackageResponse,
  GymRandomizationLevel,
  GymRewardPreset,
  GymRlAlgorithm,
  LeRobotDatasetPreviewResponse,
  LeRobotPolicyTarget,
  LeRobotRandomizationSpread,
  RegressionTestCase,
} from '../types/robotics';
import { copyTextToClipboard, downloadTextArtifact } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';
import { BundleExportView } from './BundleExportView';

export type TrainingStudioTab = 'gym_mjx' | 'lerobot' | 'deployment_bundle';

interface TrainingExportViewProps {
  session: CompilerSessionState;
  initialTab?: TrainingStudioTab;
  onRefreshBundle: () => Promise<void>;
  isRefreshingBundle: boolean;
  onNotify: (message: string) => void;
}

const JOINT_DIM_LABELS = [
  'base_yaw',
  'shoulder',
  'elbow',
  'wrist_p',
  'wrist_r',
  'flange',
  'gripper',
];

/**
 * Formats a byte size cleanly for file inspector tabs.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

/**
 * Renders the Training & Export Studio for Gymnasium/MJX RL packages,
 * Synthetic LeRobot v2.1 datasets, and Long Decode deployment bundles.
 */
export const TrainingExportView: React.FC<TrainingExportViewProps> = ({
  session,
  initialTab = 'gym_mjx',
  onRefreshBundle,
  isRefreshingBundle,
  onNotify,
}) => {
  const [activeTab, setActiveTab] = useState<TrainingStudioTab>(initialTab);
  const [savedCases, setSavedCases] = useState<RegressionTestCase[]>([]);
  const [selectedSourceKey, setSelectedSourceKey] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Gymnasium + MJX configuration state
  const [gymAlgo, setGymAlgo] = useState<GymRlAlgorithm>(APP_CONFIG.trainingExport.gymMjx.defaultAlgo);
  const [gymReward, setGymReward] = useState<GymRewardPreset>(
    APP_CONFIG.trainingExport.gymMjx.defaultRewardPreset,
  );
  const [gymDrLevel, setGymDrLevel] = useState<GymRandomizationLevel>(
    APP_CONFIG.trainingExport.gymMjx.defaultRandomizationLevel,
  );
  const [gymNumEnvs, setGymNumEnvs] = useState<number>(APP_CONFIG.trainingExport.gymMjx.defaultNumEnvs);
  const [gymTimesteps, setGymTimesteps] = useState<number>(
    APP_CONFIG.trainingExport.gymMjx.defaultTotalTimesteps,
  );
  const [gymPreview, setGymPreview] = useState<GymMjxPackageResponse | null>(null);
  const [gymSelectedFileIdx, setGymSelectedFileIdx] = useState(0);
  const [isLoadingGym, setIsLoadingGym] = useState(false);
  const [isDownloadingGym, setIsDownloadingGym] = useState(false);

  // LeRobot synthetic dataset configuration state
  const [lrEpisodeCount, setLrEpisodeCount] = useState<number>(
    APP_CONFIG.trainingExport.lerobot.defaultEpisodeCount,
  );
  const [lrFps, setLrFps] = useState<number>(APP_CONFIG.trainingExport.lerobot.defaultFps);
  const [lrPolicyTarget, setLrPolicyTarget] = useState<LeRobotPolicyTarget>(
    APP_CONFIG.trainingExport.lerobot.defaultPolicyTarget,
  );
  const [lrSpread, setLrSpread] = useState<LeRobotRandomizationSpread>(
    APP_CONFIG.trainingExport.lerobot.defaultRandomizationSpread,
  );
  const [lrIncludeFailed, setLrIncludeFailed] = useState<boolean>(false);
  const [lrPreview, setLrPreview] = useState<LeRobotDatasetPreviewResponse | null>(null);
  const [lrSubPanel, setLrSubPanel] = useState<'episodes' | 'stats' | 'files'>('episodes');
  const [lrSelectedFileIdx, setLrSelectedFileIdx] = useState(0);
  const [isLoadingLr, setIsLoadingLr] = useState(false);
  const [isDownloadingLr, setIsDownloadingLr] = useState(false);

  // Sync external initialTab changes when opened from Command Center Export menu
  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  // Load saved Regression CI cases so users can export even without a freshly compiled session
  useEffect(() => {
    let active = true;
    void fetchRegressionSuite()
      .then((snapshot) => {
        if (!active) return;
        setSavedCases(snapshot.cases);
      })
      .catch(() => {
        // Non-fatal if regression bank is unavailable
      });
    return () => {
      active = false;
    };
  }, []);

  const sourceOptions = useMemo(() => {
    const opts: Array<{ key: string; label: string; sessionId?: string; regressionCaseId?: string }> =
      [];
    if (session.sceneSpec) {
      opts.push({
        key: `session:${session.id}`,
        label: `Active Session · ${session.sceneSpec.targetObject.name} (${APP_CONFIG.robots[session.robotModel].name})`,
        sessionId: session.id,
      });
    }
    for (const c of savedCases) {
      opts.push({
        key: `case:${c.id}`,
        label: `Saved Case · ${c.name} (${APP_CONFIG.robots[c.robotModel].name})`,
        regressionCaseId: c.id,
      });
    }
    return opts;
  }, [session.id, session.robotModel, session.sceneSpec, savedCases]);

  // Default to Active Session when compiled, else first saved Regression CI case
  useEffect(() => {
    if (sourceOptions.length === 0) {
      setSelectedSourceKey('');
      return;
    }
    if (!sourceOptions.some((o) => o.key === selectedSourceKey)) {
      setSelectedSourceKey(sourceOptions[0].key);
    }
  }, [sourceOptions, selectedSourceKey]);

  const activeSource = useMemo(
    () => sourceOptions.find((o) => o.key === selectedSourceKey) ?? null,
    [sourceOptions, selectedSourceKey],
  );

  /**
   * Copies text to clipboard with brief confirmation feedback.
   */
  const handleCopy = async (key: string, text: string): Promise<void> => {
    logFunctionCall('TrainingExportView.handleCopy', { key, length: text.length });
    const ok = await copyTextToClipboard(text);
    if (ok) {
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey(null), 1800);
    }
  };

  /**
   * Builds/refreshes the Gymnasium + MuJoCo MJX RL package preview.
   */
  const refreshGymPreview = useCallback(async (): Promise<void> => {
    if (!activeSource) return;
    logFunctionCall('TrainingExportView.refreshGymPreview', {
      source: activeSource.key,
      gymAlgo,
      gymReward,
      gymDrLevel,
      gymNumEnvs,
      gymTimesteps,
    });
    setIsLoadingGym(true);
    setError(null);
    try {
      const pkg = await previewGymMjxPackage({
        sessionId: activeSource.sessionId,
        regressionCaseId: activeSource.regressionCaseId,
        algorithm: gymAlgo,
        rewardPreset: gymReward,
        randomizationLevel: gymDrLevel,
        numEnvs: gymNumEnvs,
        totalTimesteps: gymTimesteps,
      });
      setGymPreview(pkg);
      setGymSelectedFileIdx((prev) => Math.min(prev, Math.max(0, pkg.files.length - 1)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to generate Gym/MJX package.');
    } finally {
      setIsLoadingGym(false);
    }
  }, [activeSource, gymAlgo, gymReward, gymDrLevel, gymNumEnvs, gymTimesteps]);

  /**
   * Generates/refreshes the Synthetic LeRobot v2.1 episode dataset preview.
   */
  const refreshLeRobotPreview = useCallback(async (): Promise<void> => {
    if (!activeSource) return;
    logFunctionCall('TrainingExportView.refreshLeRobotPreview', {
      source: activeSource.key,
      lrEpisodeCount,
      lrFps,
      lrPolicyTarget,
      lrSpread,
      lrIncludeFailed,
    });
    setIsLoadingLr(true);
    setError(null);
    try {
      const dataset = await previewLeRobotDataset({
        sessionId: activeSource.sessionId,
        regressionCaseId: activeSource.regressionCaseId,
        episodeCount: lrEpisodeCount,
        fps: lrFps,
        policyTarget: lrPolicyTarget,
        randomizationSpread: lrSpread,
        includeFailedEpisodes: lrIncludeFailed,
      });
      setLrPreview(dataset);
      setLrSelectedFileIdx((prev) => Math.min(prev, Math.max(0, dataset.previewFiles.length - 1)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to generate LeRobot dataset.');
    } finally {
      setIsLoadingLr(false);
    }
  }, [activeSource, lrEpisodeCount, lrFps, lrPolicyTarget, lrSpread, lrIncludeFailed]);

  // Auto-generate preview whenever active tab or source/config changes
  useEffect(() => {
    if (!activeSource) return;
    if (activeTab === 'gym_mjx') {
      void refreshGymPreview();
    } else if (activeTab === 'lerobot') {
      void refreshLeRobotPreview();
    }
  }, [activeTab, activeSource, refreshGymPreview, refreshLeRobotPreview]);

  /**
   * Downloads the full Gymnasium + MuJoCo MJX `.zip` training archive.
   */
  const handleDownloadGymZip = async (): Promise<void> => {
    if (!activeSource) return;
    setIsDownloadingGym(true);
    setError(null);
    try {
      const filename = await downloadGymMjxPackageZip({
        sessionId: activeSource.sessionId,
        regressionCaseId: activeSource.regressionCaseId,
        algorithm: gymAlgo,
        rewardPreset: gymReward,
        randomizationLevel: gymDrLevel,
        numEnvs: gymNumEnvs,
        totalTimesteps: gymTimesteps,
      });
      onNotify(`Downloaded ${filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Gym/MJX ZIP download failed.');
    } finally {
      setIsDownloadingGym(false);
    }
  };

  /**
   * Downloads the complete Synthetic LeRobot v2.1 dataset `.zip` archive.
   */
  const handleDownloadLeRobotZip = async (): Promise<void> => {
    if (!activeSource) return;
    setIsDownloadingLr(true);
    setError(null);
    try {
      const filename = await downloadLeRobotDatasetZip({
        sessionId: activeSource.sessionId,
        regressionCaseId: activeSource.regressionCaseId,
        episodeCount: lrEpisodeCount,
        fps: lrFps,
        policyTarget: lrPolicyTarget,
        randomizationSpread: lrSpread,
        includeFailedEpisodes: lrIncludeFailed,
      });
      onNotify(`Downloaded ${filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'LeRobot dataset ZIP download failed.');
    } finally {
      setIsDownloadingLr(false);
    }
  };

  const gymActiveFile = gymPreview?.files[gymSelectedFileIdx] ?? null;
  const lrActiveFile = lrPreview?.previewFiles[lrSelectedFileIdx] ?? null;

  return (
    <div className="flex flex-col gap-5">
      {/* Studio Mode Switcher & Calibrated Source Selector */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
          <button
            type="button"
            onClick={() => {
              logFunctionCall('TrainingExportView.selectTab', { tab: 'gym_mjx' });
              setActiveTab('gym_mjx');
            }}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'gym_mjx'
                ? 'bg-amber-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Gymnasium + MJX RL</span>
          </button>

          <button
            type="button"
            onClick={() => {
              logFunctionCall('TrainingExportView.selectTab', { tab: 'lerobot' });
              setActiveTab('lerobot');
            }}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'lerobot'
                ? 'bg-amber-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>LeRobot Dataset</span>
          </button>

          <button
            type="button"
            onClick={() => {
              logFunctionCall('TrainingExportView.selectTab', { tab: 'deployment_bundle' });
              setActiveTab('deployment_bundle');
            }}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'deployment_bundle'
                ? 'bg-amber-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Package className="w-3.5 h-3.5" />
            <span>Deployment Bundle</span>
          </button>
        </div>

        {activeTab !== 'deployment_bundle' && (
          <div className="flex items-center gap-2.5 min-w-0">
            <label
              htmlFor="training-source-select"
              className="text-xs text-slate-400 whitespace-nowrap"
            >
              Calibrated Scene
            </label>
            <select
              id="training-source-select"
              value={selectedSourceKey}
              onChange={(e) => setSelectedSourceKey(e.target.value)}
              disabled={sourceOptions.length === 0}
              className="max-w-[340px] bg-[#0B0E17] border border-[#2D3B52] rounded-md px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-amber-500"
            >
              {sourceOptions.length === 0 ? (
                <option value="">No compiled scene yet</option>
              ) : (
                sourceOptions.map((opt) => (
                  <option key={opt.key} value={opt.key}>
                    {opt.label}
                  </option>
                ))
              )}
            </select>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-rose-950/40 border border-rose-500/40 rounded-lg px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* TAB 3: Existing Long Decode Deployment Bundle */}
      {activeTab === 'deployment_bundle' && (
        <BundleExportView
          bundle={session.bundle}
          onRefreshBundle={onRefreshBundle}
          isRefreshing={isRefreshingBundle}
        />
      )}

      {/* Empty State when no Active Session or Saved Regression Case exists */}
      {activeTab !== 'deployment_bundle' && !activeSource && (
        <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-10 text-center">
          <h2 className="font-display text-lg font-semibold text-slate-100">
            Compile a Scene to Generate Training Exports
          </h2>
          <p className="mt-2 text-sm text-slate-400 max-w-lg mx-auto">
            Upload a tabletop video in the Command Center and run the Gemini 4 pipeline (or save a case in Regression CI) to export runnable Gymnasium/MuJoCo MJX RL environments and synthetic LeRobot v2.1 episode datasets.
          </p>
        </div>
      )}

      {/* TAB 1: Gymnasium + MuJoCo MJX RL Training Package */}
      {activeTab === 'gym_mjx' && activeSource && (
        <div className="flex flex-col gap-5">
          {/* RL Configuration & Primary ZIP CTA */}
          <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-5 flex flex-col gap-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="font-display text-base font-semibold text-slate-100">
                  Exportable Gymnasium + MuJoCo MJX RL Training Package
                </h2>
                <p className="mt-1 text-xs text-slate-400 max-w-2xl">
                  Generates a self-contained RL codebase calibrated to your scene&apos;s exact object dimensions, friction, mass, obstacles, and Goal Placement Pad — ready for local Stable-Baselines3 or GPU-vectorized JAX MJX training.
                </p>
              </div>

              <button
                type="button"
                disabled={isDownloadingGym || isLoadingGym}
                onClick={() => void handleDownloadGymZip()}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 rounded-md hover:bg-amber-400 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <FileArchive className="w-4 h-4" />
                <span>{isDownloadingGym ? 'Packaging ZIP...' : 'Download RL Package (.zip)'}</span>
              </button>
            </div>

            {/* Parameter Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 pt-3 border-t border-[#1E293B]">
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">RL Algorithm</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {(
                    [
                      { id: 'ppo', label: 'PPO' },
                      { id: 'sac', label: 'SAC' },
                      { id: 'mjx_ppo', label: 'MJX' },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setGymAlgo(item.id)}
                      className={`py-1.5 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                        gymAlgo === item.id
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Reward Shaping</span>
                <select
                  value={gymReward}
                  onChange={(e) => setGymReward(e.target.value as GymRewardPreset)}
                  className="bg-[#0B0E17] border border-[#1E293B] rounded-md px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-amber-500"
                >
                  <option value="dense_pick_place">Dense Pick &amp; Place</option>
                  <option value="force_safe_precision">Force-Safe Precision</option>
                  <option value="sparse_goal">Sparse Goal Completion</option>
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Domain Randomization</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {(
                    [
                      { id: 'off', label: 'Off' },
                      { id: 'standard', label: 'Std' },
                      { id: 'aggressive', label: 'High' },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setGymDrLevel(item.id)}
                      className={`py-1.5 text-xs rounded transition-colors cursor-pointer ${
                        gymDrLevel === item.id
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Parallel Envs</span>
                <div className="grid grid-cols-4 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {APP_CONFIG.trainingExport.gymMjx.numEnvsPresets.map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setGymNumEnvs(n)}
                      className={`py-1.5 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                        gymNumEnvs === n
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Training Budget</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {APP_CONFIG.trainingExport.gymMjx.timestepPresets.map((steps) => (
                    <button
                      key={steps}
                      type="button"
                      onClick={() => setGymTimesteps(steps)}
                      className={`py-1.5 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                        gymTimesteps === steps
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {steps >= 1_000_000 ? `${steps / 1_000_000}M` : `${steps / 1000}k`}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {gymPreview && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono-tabular text-slate-400 pt-2 border-t border-[#1E293B]">
                <span className="text-slate-200">{gymPreview.packageName}.zip</span>
                <span aria-hidden="true">·</span>
                <span>{gymPreview.files.length} files</span>
                <span aria-hidden="true">·</span>
                <span>{gymPreview.config.controlHz} Hz control</span>
                <span aria-hidden="true">·</span>
                <span>
                  Goal Pad ({gymPreview.goalPad.x.toFixed(2)}m, {gymPreview.goalPad.z.toFixed(2)}m)
                </span>
                <span aria-hidden="true">·</span>
                <span>Obs 29-D / Act 7-D continuous</span>
              </div>
            )}
          </div>

          {/* 6-File Code Inspector */}
          <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
            <div className="px-4 py-2.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                {(gymPreview?.files ?? []).map((file, idx) => (
                  <button
                    key={file.path}
                    type="button"
                    onClick={() => setGymSelectedFileIdx(idx)}
                    className={`px-2.5 py-1 text-xs font-mono-tabular rounded transition-colors whitespace-nowrap cursor-pointer ${
                      gymSelectedFileIdx === idx
                        ? 'bg-[#1E293B] text-white font-medium'
                        : 'text-slate-400 hover:text-slate-100'
                    }`}
                  >
                    {file.path}
                  </button>
                ))}
              </div>

              {gymActiveFile && (
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono-tabular text-slate-400">
                    {formatBytes(gymActiveFile.sizeBytes)}
                  </span>
                  <button
                    type="button"
                    onClick={() => void handleCopy(`gym:${gymActiveFile.path}`, gymActiveFile.content)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-[#161D2C] border border-[#1E293B] rounded hover:text-white cursor-pointer"
                  >
                    {copiedKey === `gym:${gymActiveFile.path}` ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>{copiedKey === `gym:${gymActiveFile.path}` ? 'Copied' : 'Copy'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => downloadTextArtifact(gymActiveFile.path, gymActiveFile.content)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-cyan-300 bg-[#161D2C] border border-[#1E293B] rounded hover:border-cyan-500/40 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download File</span>
                  </button>
                </div>
              )}
            </div>

            <pre className="p-4 overflow-auto max-h-[500px] bg-[#0B0E17] font-mono text-xs leading-6 text-slate-200 whitespace-pre">
              {isLoadingGym && !gymActiveFile
                ? 'Generating Gymnasium + MuJoCo MJX package...'
                : gymActiveFile?.content ?? ''}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 2: Synthetic LeRobot v2.1 Episode Dataset Generator */}
      {activeTab === 'lerobot' && activeSource && (
        <div className="flex flex-col gap-5">
          {/* Dataset Generator Controls & Primary ZIP CTA */}
          <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-5 flex flex-col gap-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="font-display text-base font-semibold text-slate-100">
                  Synthetic LeRobot v2.1 Episode Dataset Generator
                </h2>
                <p className="mt-1 text-xs text-slate-400 max-w-2xl">
                  Rolls out randomized pick-lift-transport-place trajectories through your calibrated physics scene, filters out failed episodes, computes LeRobot normalization stats, and exports a ready-to-train dataset for ACT, Diffusion Policy, or SmolVLA.
                </p>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  disabled={isLoadingLr}
                  onClick={() => void refreshLeRobotPreview()}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded-md hover:text-white disabled:opacity-50 transition-colors cursor-pointer"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isLoadingLr ? 'animate-spin' : ''}`} />
                  <span>{isLoadingLr ? 'Rolling Out...' : 'Regenerate'}</span>
                </button>
                <button
                  type="button"
                  disabled={isDownloadingLr || isLoadingLr}
                  onClick={() => void handleDownloadLeRobotZip()}
                  className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 rounded-md hover:bg-amber-400 disabled:opacity-50 transition-colors whitespace-nowrap cursor-pointer"
                >
                  <FileArchive className="w-4 h-4" />
                  <span>
                    {isDownloadingLr ? 'Building Dataset ZIP...' : 'Download LeRobot Dataset (.zip)'}
                  </span>
                </button>
              </div>
            </div>

            {/* Dataset Parameter Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 pt-3 border-t border-[#1E293B]">
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Synthetic Episodes</span>
                <div className="grid grid-cols-4 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {APP_CONFIG.trainingExport.lerobot.episodeCountPresets.map((count) => (
                    <button
                      key={count}
                      type="button"
                      onClick={() => setLrEpisodeCount(count)}
                      className={`py-1.5 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                        lrEpisodeCount === count
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {count}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Sampling Rate</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {APP_CONFIG.trainingExport.lerobot.fpsPresets.map((hz) => (
                    <button
                      key={hz}
                      type="button"
                      onClick={() => setLrFps(hz)}
                      className={`py-1.5 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                        lrFps === hz
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {hz} Hz
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Target Policy</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {(
                    [
                      { id: 'act', label: 'ACT' },
                      { id: 'diffusion', label: 'Diff' },
                      { id: 'smolvla', label: 'SmolVLA' },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setLrPolicyTarget(item.id)}
                      className={`py-1.5 text-xs rounded transition-colors cursor-pointer ${
                        lrPolicyTarget === item.id
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-400">Randomization Spread</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {(
                    [
                      { id: 'low', label: 'Low' },
                      { id: 'medium', label: 'Med' },
                      { id: 'high', label: 'High' },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setLrSpread(item.id)}
                      className={`py-1.5 text-xs rounded transition-colors cursor-pointer ${
                        lrSpread === item.id
                          ? 'bg-[#1E293B] text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col justify-end pb-1">
                <label className="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={!lrIncludeFailed}
                    onChange={(e) => setLrIncludeFailed(!e.target.checked)}
                    className="rounded border-[#2D3B52] bg-[#0B0E17] accent-amber-500"
                  />
                  <span>Verified passes only</span>
                </label>
              </div>
            </div>
          </div>

          {/* 4-Column Dataset Telemetry Strip */}
          {lrPreview && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4">
                <div className="text-xs text-slate-400">Exported Episodes</div>
                <div className="mt-1 font-mono-tabular text-2xl font-bold text-slate-100">
                  {lrPreview.exportedEpisodes}
                  <span className="text-xs font-normal text-slate-400 ml-1.5">
                    / {lrPreview.attemptedEpisodes} tried
                  </span>
                </div>
                <div className="mt-1 text-xs font-mono-tabular text-emerald-400">
                  {lrPreview.successRatePct}% pass · {lrPreview.filteredOutEpisodes} filtered
                </div>
              </div>

              <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4">
                <div className="text-xs text-slate-400">Total Synchronized Frames</div>
                <div className="mt-1 font-mono-tabular text-2xl font-bold text-slate-100">
                  {lrPreview.totalFrames.toLocaleString()}
                  <span className="text-xs font-normal text-slate-400 ml-1.5">frames</span>
                </div>
                <div className="mt-1 text-xs font-mono-tabular text-slate-400">
                  LeRobot {lrPreview.codebaseVersion} · {lrPreview.fps} Hz
                </div>
              </div>

              <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4">
                <div className="text-xs text-slate-400">Mean Peak Contact Force</div>
                <div className="mt-1 font-mono-tabular text-2xl font-bold text-slate-100">
                  {lrPreview.meanPeakForceN.toFixed(1)}
                  <span className="text-xs font-normal text-slate-400 ml-1.5">N</span>
                </div>
                <div className="mt-1 text-xs font-mono-tabular text-slate-400">
                  Ceiling {APP_CONFIG.simulation.maxSafeContactForceN.toFixed(0)} N
                </div>
              </div>

              <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4">
                <div className="text-xs text-slate-400">Mean Goal Pad Error</div>
                <div className="mt-1 font-mono-tabular text-2xl font-bold text-emerald-400">
                  {lrPreview.meanPlacementErrorMm.toFixed(1)}
                  <span className="text-xs font-normal text-slate-400 ml-1.5">mm</span>
                </div>
                <div className="mt-1 text-xs font-mono-tabular text-slate-400">
                  Target · {lrPreview.policyTarget.toUpperCase()}
                </div>
              </div>
            </div>
          )}

          {/* Dataset Inspector Sub-Tabs */}
          {lrPreview && (
            <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
              <div className="px-4 py-2.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
                  {(
                    [
                      { id: 'episodes', label: `Episodes (${lrPreview.exportedEpisodes})` },
                      { id: 'stats', label: 'Normalization Stats' },
                      { id: 'files', label: 'Metadata & Training Scripts' },
                    ] as const
                  ).map((sub) => (
                    <button
                      key={sub.id}
                      type="button"
                      onClick={() => setLrSubPanel(sub.id)}
                      className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                        lrSubPanel === sub.id
                          ? 'bg-[#1E293B] text-white'
                          : 'text-slate-400 hover:text-slate-100'
                      }`}
                    >
                      {sub.label}
                    </button>
                  ))}
                </div>

                <div className="text-xs font-mono-tabular text-slate-400">
                  repo_id: <span className="text-slate-200">{lrPreview.repoIdSuggestion}</span>
                </div>
              </div>

              {/* Sub-Panel A: Episode Rollouts Table */}
              {lrSubPanel === 'episodes' && (
                <div className="overflow-x-auto max-h-[420px]">
                  <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-[#0B0E17] border-b border-[#1E293B] text-xs text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5 font-medium">Episode</th>
                        <th className="px-4 py-2.5 font-medium">Status</th>
                        <th className="px-4 py-2.5 font-medium">Frames</th>
                        <th className="px-4 py-2.5 font-medium">Friction μ</th>
                        <th className="px-4 py-2.5 font-medium">Mass</th>
                        <th className="px-4 py-2.5 font-medium">XY Jitter (cm)</th>
                        <th className="px-4 py-2.5 font-medium">Peak Force</th>
                        <th className="px-4 py-2.5 font-medium">Pad Error</th>
                        <th className="px-4 py-2.5 font-medium">Return</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E293B] text-xs font-mono-tabular">
                      {lrPreview.episodes.map((ep) => (
                        <tr key={ep.episodeIndex} className="hover:bg-[#161D2C]/60">
                          <td className="px-4 py-2 text-slate-200">
                            ep_{String(ep.episodeIndex).padStart(4, '0')}
                          </td>
                          <td
                            className={`px-4 py-2 font-medium ${
                              ep.passed ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {ep.passed ? '✓ Pass' : '✕ Fail'}
                          </td>
                          <td className="px-4 py-2 text-slate-300">{ep.lengthFrames}</td>
                          <td className="px-4 py-2 text-slate-300">{ep.friction.toFixed(2)}</td>
                          <td className="px-4 py-2 text-slate-300">{ep.massKg.toFixed(3)} kg</td>
                          <td className="px-4 py-2 text-slate-400">
                            {(ep.xyOffset[0] * 100).toFixed(1)}, {(ep.xyOffset[1] * 100).toFixed(1)}
                          </td>
                          <td className="px-4 py-2 text-slate-300">
                            {ep.peakContactForceN.toFixed(1)} N
                          </td>
                          <td className="px-4 py-2 text-slate-300">
                            {(ep.placementErrorM * 1000).toFixed(1)} mm
                          </td>
                          <td className="px-4 py-2 text-cyan-300">{ep.totalReward.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Sub-Panel B: Per-Feature Normalization Stats */}
              {lrSubPanel === 'stats' && (
                <div className="p-4 flex flex-col gap-4">
                  <p className="text-xs text-slate-400">
                    Exact dataset-wide normalization statistics written into{' '}
                    <span className="font-mono-tabular text-slate-200">
                      meta/episodes_stats.jsonl
                    </span>{' '}
                    for ACT / Diffusion Policy / SmolVLA input &amp; action normalization.
                  </p>
                  {lrPreview.featureStats.slice(0, 2).map((stat) => (
                    <div key={stat.featureName} className="overflow-x-auto border border-[#1E293B] rounded-md">
                      <div className="px-3 py-2 bg-[#0B0E17] border-b border-[#1E293B] flex items-center justify-between text-xs font-mono-tabular">
                        <span className="text-slate-100 font-semibold">{stat.featureName}</span>
                        <span className="text-slate-400">shape [{stat.shape.join(', ')}]</span>
                      </div>
                      <table className="w-full text-left border-collapse text-xs font-mono-tabular">
                        <thead className="bg-[#0B0E17]/60 text-slate-400 border-b border-[#1E293B]">
                          <tr>
                            <th className="px-3 py-1.5 font-medium">Channel</th>
                            <th className="px-3 py-1.5 font-medium">Min</th>
                            <th className="px-3 py-1.5 font-medium">Mean</th>
                            <th className="px-3 py-1.5 font-medium">Std</th>
                            <th className="px-3 py-1.5 font-medium">Max</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#1E293B]">
                          {stat.mean.map((meanVal, idx) => (
                            <tr key={idx}>
                              <td className="px-3 py-1.5 text-slate-200">
                                {JOINT_DIM_LABELS[idx] ?? `dim_${idx}`}
                              </td>
                              <td className="px-3 py-1.5 text-slate-400">
                                {(stat.min[idx] ?? 0).toFixed(4)}
                              </td>
                              <td className="px-3 py-1.5 text-slate-200">{meanVal.toFixed(4)}</td>
                              <td className="px-3 py-1.5 text-cyan-300">
                                {(stat.std[idx] ?? 0).toFixed(4)}
                              </td>
                              <td className="px-3 py-1.5 text-slate-400">
                                {(stat.max[idx] ?? 0).toFixed(4)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}

              {/* Sub-Panel C: Metadata & Training Scripts Preview */}
              {lrSubPanel === 'files' && (
                <div>
                  <div className="px-4 py-2.5 bg-[#0B0E17] border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-1">
                      {lrPreview.previewFiles.map((file, idx) => (
                        <button
                          key={file.path}
                          type="button"
                          onClick={() => setLrSelectedFileIdx(idx)}
                          className={`px-2.5 py-1 text-xs font-mono-tabular rounded transition-colors whitespace-nowrap cursor-pointer ${
                            lrSelectedFileIdx === idx
                              ? 'bg-[#1E293B] text-white font-medium'
                              : 'text-slate-400 hover:text-slate-100'
                          }`}
                        >
                          {file.path}
                        </button>
                      ))}
                    </div>

                    {lrActiveFile && (
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() =>
                            void handleCopy(`lr:${lrActiveFile.path}`, lrActiveFile.content)
                          }
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-[#161D2C] border border-[#1E293B] rounded hover:text-white cursor-pointer"
                        >
                          {copiedKey === `lr:${lrActiveFile.path}` ? (
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                          <span>{copiedKey === `lr:${lrActiveFile.path}` ? 'Copied' : 'Copy'}</span>
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            downloadTextArtifact(
                              lrActiveFile.path.split('/').pop() ?? lrActiveFile.path,
                              lrActiveFile.content,
                            )
                          }
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-cyan-300 bg-[#161D2C] border border-[#1E293B] rounded hover:border-cyan-500/40 cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5" />
                          <span>Download File</span>
                        </button>
                      </div>
                    )}
                  </div>

                  <pre className="p-4 overflow-auto max-h-[460px] bg-[#0B0E17] font-mono text-xs leading-6 text-slate-200 whitespace-pre">
                    {lrActiveFile?.content ?? ''}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
