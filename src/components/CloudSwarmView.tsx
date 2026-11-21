/**
 * External Cloud MuJoCo Swarm Webhook Connector (`CloudSwarmView`)
 *
 * Configures how the 20-environment domain-randomized physics swarm executes
 * during each closed-loop Gemini 4 repair iteration:
 * - Built-In Server Physics Evaluator (zero setup, deterministic rigid-body
 *   kinematics & contact-force engine running locally on Express).
 * - External Cloud MuJoCo Webhook (connects directly to a user's deployed
 *   HTTPS Modal or Google Cloud Run headless OSMesa/EGL MuJoCo cluster, with
 *   server-side outbound URL validation and live reachability testing).
 *
 * Use cases:
 * - Plugging in a live 20-container Modal MuJoCo worker during a hackathon demo.
 * - Verifying webhook latency and copying the Modal FastAPI wrapper schema.
 */

import React, { useState } from 'react';
import { Check, Copy, Server, Wifi } from 'lucide-react';
import type { WebhookConfig } from '../types/robotics';
import { copyTextToClipboard } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

interface CloudSwarmViewProps {
  webhook: WebhookConfig;
  onSaveWebhook: (params: { enabled: boolean; url: string; authToken?: string }) => Promise<void>;
  onTestWebhook: (url: string, authToken?: string) => Promise<{ ok: boolean; latencyMs: number; message: string }>;
}

const MODAL_WRAPPER_SNIPPET = `# Modal HTTPS Web Endpoint for LetThereBeSims (cloud_sim/modal_worker.py)
@app.function(image=sim_image, cpu=2, memory=2048, timeout=120)
@modal.fastapi_endpoint(method="POST")
async def swarm_webhook(payload: dict):
    if payload.get("healthcheck"):
        return {"ok": True, "engine": "mujoco-3.2-osmesa"}
    return await run_cloud_swarm(
        mjcf_xml=payload["mjcf_xml"],
        randomizations=payload["randomizations"],
        controller_code=payload["controller_code"],
    )`;

/**
 * Renders the External Cloud MuJoCo Webhook configuration, reachability tester,
 * and Modal endpoint integration contract.
 */
export const CloudSwarmView: React.FC<CloudSwarmViewProps> = ({
  webhook,
  onSaveWebhook,
  onTestWebhook,
}) => {
  const [enabled, setEnabled] = useState(webhook.enabled);
  const [url, setUrl] = useState(webhook.url);
  const [authToken, setAuthToken] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState(false);

  /**
   * Executes a server-side SSRF-validated health probe against the entered URL.
   */
  const handleTest = async (): Promise<void> => {
    logFunctionCall('CloudSwarmView.handleTest', { url });
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await onTestWebhook(url, authToken || undefined);
      setTestResult({ ok: res.ok, message: res.message });
    } catch (err) {
      setTestResult({
        ok: false,
        message: err instanceof Error ? err.message : 'Test request failed.',
      });
    } finally {
      setIsTesting(false);
    }
  };

  /**
   * Persists the selected swarm execution mode and validated HTTPS URL on the
   * current Express session.
   */
  const handleSave = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    logFunctionCall('CloudSwarmView.handleSave', { enabled, url });
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await onSaveWebhook({
        enabled,
        url,
        authToken: authToken || undefined,
      });
      setSaveMessage(
        enabled
          ? 'External Cloud MuJoCo webhook enabled for upcoming pipeline runs.'
          : 'Switched to Built-In 20-Environment Physics Evaluator.',
      );
      setAuthToken('');
    } catch (err) {
      setSaveMessage(
        err instanceof Error ? err.message : 'Failed to save webhook configuration.',
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <form
        onSubmit={(e) => void handleSave(e)}
        className="lg:col-span-7 bg-[#111622] border border-[#1E293B] rounded-lg p-5 flex flex-col gap-5"
      >
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-[#1E293B]">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-100">
              20-Environment Swarm Execution Backend
            </h2>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">
              Route parallel MuJoCo rollouts to your deployed Modal or Google Cloud Run worker, or run the built-in rigid-body physics evaluator with zero external setup.
            </p>
          </div>
          <Server className="w-5 h-5 text-cyan-400 shrink-0" />
        </div>

        {/* Execution Mode Segmented Switch */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setEnabled(false)}
            className={`p-3.5 text-left rounded-md border transition-colors cursor-pointer ${
              !enabled
                ? 'bg-[#161D2C] border-amber-500'
                : 'bg-[#0B0E17] border-[#1E293B] hover:border-[#2D3B52]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-100">
                Built-In Physics Evaluator
              </span>
              <span className="font-mono-tabular text-[11px] text-emerald-400">
                {!enabled ? '✓ Active' : 'Standby'}
              </span>
            </div>
            <p className="mt-1.5 text-[11px] text-slate-400 leading-relaxed">
              Evaluates 20 randomized mass, friction, and XY perturbations locally on the server with 60-frame Three.js IK replay.
            </p>
          </button>

          <button
            type="button"
            onClick={() => setEnabled(true)}
            className={`p-3.5 text-left rounded-md border transition-colors cursor-pointer ${
              enabled
                ? 'bg-[#161D2C] border-amber-500'
                : 'bg-[#0B0E17] border-[#1E293B] hover:border-[#2D3B52]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-100">
                External Cloud MuJoCo Webhook
              </span>
              <span className="font-mono-tabular text-[11px] text-cyan-400">
                {enabled ? '✓ Active' : 'Optional'}
              </span>
            </div>
            <p className="mt-1.5 text-[11px] text-slate-400 leading-relaxed">
              Dispatches MJCF XML + Python policy to your HTTPS Modal or Cloud Run OSMesa swarm and streams rendered MP4 clips back.
            </p>
          </button>
        </div>

        {/* Webhook URL & Optional Bearer Header */}
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="webhook-url" className="text-xs font-medium text-slate-300">
              HTTPS Swarm Endpoint URL
            </label>
            <input
              id="webhook-url"
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://your-workspace--gemini4-sim2real-swarm-webhook.modal.run"
              className="w-full px-3 py-2 text-xs font-mono text-slate-100 bg-[#0B0E17] border border-[#1E293B] rounded-md focus:outline-none focus:border-amber-500"
            />
            <span className="text-[11px] text-slate-500">
              Enforces HTTPS and blocks loopback, private network, and metadata addresses automatically.
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="webhook-token" className="text-xs font-medium text-slate-300">
              Bearer Token (Optional — held in server session memory only)
            </label>
            <input
              id="webhook-token"
              type="password"
              value={authToken}
              onChange={(e) => setAuthToken(e.target.value)}
              placeholder={
                webhook.hasAuthToken
                  ? '•••••••• (token configured on server; leave blank to keep)'
                  : 'Optional Authorization bearer token'
              }
              className="w-full px-3 py-2 text-xs font-mono text-slate-100 bg-[#0B0E17] border border-[#1E293B] rounded-md focus:outline-none focus:border-amber-500"
            />
          </div>
        </div>

        {testResult && (
          <div
            className={`p-3 rounded-md border text-xs ${
              testResult.ok
                ? 'bg-emerald-500/10 border-emerald-500/35 text-emerald-200'
                : 'bg-rose-500/10 border-rose-500/35 text-rose-200'
            }`}
          >
            {testResult.message}
          </div>
        )}

        {saveMessage && (
          <p className="text-xs text-amber-300">{saveMessage}</p>
        )}

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <button
            type="button"
            disabled={isTesting || !url.trim()}
            onClick={() => void handleTest()}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#1E293B] rounded-md hover:border-[#2D3B52] disabled:opacity-50 transition-colors cursor-pointer"
          >
            <Wifi className="w-3.5 h-3.5" />
            <span>{isTesting ? 'Probing Endpoint...' : 'Test Connection'}</span>
          </button>
          <button
            type="submit"
            disabled={isSaving}
            className="px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 rounded-md hover:bg-amber-400 disabled:opacity-50 transition-colors cursor-pointer"
          >
            {isSaving ? 'Saving...' : 'Save Swarm Configuration'}
          </button>
        </div>
      </form>

      {/* Right Reference Contract for Modal / Cloud Run */}
      <div className="lg:col-span-5 bg-[#111622] border border-[#1E293B] rounded-lg p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="font-display text-sm font-semibold text-slate-100">
            Modal / Cloud Run Worker Contract
          </h3>
          <button
            type="button"
            onClick={async () => {
              const ok = await copyTextToClipboard(MODAL_WRAPPER_SNIPPET);
              if (ok) {
                setCopiedSnippet(true);
                window.setTimeout(() => setCopiedSnippet(false), 1800);
              }
            }}
            className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-slate-300 bg-[#161D2C] border border-[#1E293B] rounded hover:text-white cursor-pointer"
          >
            {copiedSnippet ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            <span>{copiedSnippet ? 'Copied' : 'Copy Snippet'}</span>
          </button>
        </div>

        <pre className="p-3.5 rounded-md bg-[#0B0E17] border border-[#1E293B] font-mono text-[11px] leading-5 text-slate-200 overflow-x-auto">
          {MODAL_WRAPPER_SNIPPET}
        </pre>

        <div className="flex flex-col gap-2 text-xs text-slate-400 leading-relaxed">
          <p className="font-medium text-slate-200">Expected JSON Response Fields</p>
          <p>
            Return <code className="font-mono text-cyan-300">rollouts[]</code> containing{' '}
            <code className="font-mono text-slate-200">env_id</code>,{' '}
            <code className="font-mono text-slate-200">success</code>,{' '}
            <code className="font-mono text-slate-200">final_object_z</code>,{' '}
            <code className="font-mono text-slate-200">peak_contact_force_n</code>, and optional{' '}
            <code className="font-mono text-slate-200">video_b64</code> /{' '}
            <code className="font-mono text-slate-200">preview_videos_b64</code> for direct Cloud MP4 playback inside the 3D viewport.
          </p>
        </div>
      </div>
    </div>
  );
};
