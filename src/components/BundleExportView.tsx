/**
 * Long Decode Deployment Bundle Exporter (`BundleExportView`)
 *
 * Displays and exports the production deployment package compiled asynchronously
 * by Gemini 4 Neon (`background=True`, Long Decode) at Stage 4 of the pipeline:
 * 1. Standalone LeRobot / MuJoCo Python controller with safety torque limits.
 * 2. Verified 20-environment MJCF regression test suite XML.
 * 3. Comprehensive Sim-to-Real Verification & Contact Dynamics Audit Report.
 *
 * Use cases:
 * - Monitoring background Long Decode compilation progress and refreshing on demand.
 * - Exporting individual verified `.py`, `.xml`, and `.md` deployment files.
 */

import React, { useState } from 'react';
import { Check, Copy, Download, RefreshCw } from 'lucide-react';
import type { DeploymentBundle } from '../types/robotics';
import { copyTextToClipboard, downloadTextArtifact } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

interface BundleExportViewProps {
  bundle: DeploymentBundle | null;
  onRefreshBundle: () => Promise<void>;
  isRefreshing: boolean;
}

type BundleTab = 'controller' | 'mjcf_suite' | 'audit_report';

/**
 * Renders the Long Decode bundle viewer with live refresh and artifact exports.
 */
export const BundleExportView: React.FC<BundleExportViewProps> = ({
  bundle,
  onRefreshBundle,
  isRefreshing,
}) => {
  const [activeTab, setActiveTab] = useState<BundleTab>('controller');
  const [copied, setCopied] = useState(false);

  if (!bundle) {
    return (
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-10 text-center">
        <h2 className="font-display text-lg font-semibold text-slate-100">
          Deployment Bundle Pending Verification
        </h2>
        <p className="mt-2 text-sm text-slate-400 max-w-lg mx-auto">
          Complete a closed-loop Gemini 4 policy compilation in the Command Center to trigger the asynchronous Long Decode deployment package.
        </p>
      </div>
    );
  }

  const contentMap: Record<BundleTab, { filename: string; content: string; label: string }> = {
    controller: {
      filename: 'lerobot_verified_controller.py',
      content: bundle.standaloneControllerPy,
      label: 'Standalone Controller (.py)',
    },
    mjcf_suite: {
      filename: 'mujoco_regression_suite.xml',
      content: bundle.regressionSuiteXml,
      label: 'MJCF Regression Suite (.xml)',
    },
    audit_report: {
      filename: 'lettherebesims_contact_dynamics_audit.md',
      content: bundle.auditReportMarkdown,
      label: 'Verification Audit (.md)',
    },
  };

  const currentArtifact = contentMap[activeTab];

  return (
    <div className="flex flex-col gap-5">
      {/* Header & Long Decode Status */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <h2 className="font-display text-base font-semibold text-slate-100">
              Verified Deployment Bundle (Long Decode)
            </h2>
            <span
              className={`font-mono-tabular text-xs font-medium ${
                bundle.status === 'completed'
                  ? 'text-emerald-400'
                  : bundle.status === 'failed'
                    ? 'text-rose-400'
                    : 'text-amber-400'
              }`}
            >
              {bundle.status === 'completed'
                ? '✓ Compiled'
                : bundle.status === 'failed'
                  ? '✕ Failed'
                  : '▸ Generating in Background'}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono-tabular text-slate-400">
            <span>interaction_id: {bundle.interactionId}</span>
            <span aria-hidden="true">·</span>
            <span>Updated {new Date(bundle.updatedAt).toLocaleTimeString()}</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={isRefreshing}
            onClick={() => {
              logFunctionCall('BundleExportView.refreshBundle');
              void onRefreshBundle();
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded hover:border-[#2D3B52] disabled:opacity-50 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>{isRefreshing ? 'Polling...' : 'Refresh Status'}</span>
          </button>
          <button
            type="button"
            onClick={() => {
              downloadTextArtifact(contentMap.controller.filename, contentMap.controller.content);
              downloadTextArtifact(contentMap.mjcf_suite.filename, contentMap.mjcf_suite.content);
              downloadTextArtifact(contentMap.audit_report.filename, contentMap.audit_report.content);
            }}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-amber-500 rounded hover:bg-amber-400 transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download All 3 Files</span>
          </button>
        </div>
      </div>

      {/* Tabbed Artifact Preview */}
      <div className="bg-[#111622] border border-[#1E293B] rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1 p-1 bg-[#0B0E17] border border-[#1E293B] rounded-md">
            {(Object.keys(contentMap) as BundleTab[]).map((tabKey) => (
              <button
                key={tabKey}
                type="button"
                onClick={() => {
                  logFunctionCall('BundleExportView.selectTab', { tab: tabKey });
                  setActiveTab(tabKey);
                }}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
                  activeTab === tabKey
                    ? 'bg-[#1E293B] text-white'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {contentMap[tabKey].label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={async () => {
                const ok = await copyTextToClipboard(currentArtifact.content);
                if (ok) {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1800);
                }
              }}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-[#161D2C] border border-[#1E293B] rounded hover:text-white cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              type="button"
              onClick={() =>
                downloadTextArtifact(currentArtifact.filename, currentArtifact.content)
              }
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-cyan-300 bg-[#161D2C] border border-[#1E293B] rounded hover:border-cyan-500/40 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>{currentArtifact.filename}</span>
            </button>
          </div>
        </div>

        <pre className="p-4 overflow-auto max-h-[540px] bg-[#0B0E17] font-mono text-xs leading-6 text-slate-200 whitespace-pre-wrap">
          {currentArtifact.content}
        </pre>
      </div>
    </div>
  );
};
