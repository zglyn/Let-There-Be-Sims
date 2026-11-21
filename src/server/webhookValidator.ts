/**
 * External Cloud MuJoCo Webhook Validator & Executor
 *
 * Enables robotics teams running a live Modal or Google Cloud Run headless
 * MuJoCo swarm worker (`evaluate_cloud_env`) to connect their endpoint directly
 * to the Sim2Real Compiler. Enforces strict outbound URL validation so that
 * user-supplied webhook URLs cannot target loopback, private networks, or cloud
 * metadata addresses before dispatching parallel rollout requests.
 *
 * Use cases:
 * - Testing connectivity to a deployed Modal / Cloud Run MuJoCo swarm worker.
 * - Dispatching 20 domain-randomized environments + MJCF XML + Python policy
 *   to external headless MuJoCo containers and normalizing returned telemetry
 *   and rendered MP4 preview clips for the desktop UI.
 */

import dns from 'dns/promises';
import net from 'net';
import { APP_CONFIG } from '../config/appConfig';
import type {
  ControllerParams,
  DomainRandomizationParam,
  RolloutResult,
  SceneSpec,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { evaluateBuiltInSwarm } from './physicsEvaluator';
import type { RobotModelId } from '../config/appConfig';

export interface WebhookValidationResult {
  valid: boolean;
  normalizedUrl?: string;
  error?: string;
}

/**
 * Checks whether an IPv4 or IPv6 address belongs to loopback, link-local,
 * cloud metadata, or private RFC1918/ULA ranges.
 */
function isPrivateOrRestrictedIp(ip: string): boolean {
  if (net.isIPv4(ip)) {
    const parts = ip.split('.').map(Number);
    const [a, b] = parts;
    if (a === 10) return true; // 10.0.0.0/8
    if (a === 127) return true; // 127.0.0.0/8 loopback
    if (a === 0) return true; // 0.0.0.0/8
    if (a === 169 && b === 254) return true; // 169.254.0.0/16 link-local & cloud metadata
    if (a === 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12
    if (a === 192 && b === 168) return true; // 192.168.0.0/16
    if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT
    return false;
  }
  if (net.isIPv6(ip)) {
    const lower = ip.toLowerCase();
    if (
      lower === '::1' ||
      lower === '::' ||
      lower.startsWith('fc') ||
      lower.startsWith('fd') ||
      lower.startsWith('fe80:') ||
      lower.startsWith('::ffff:127.') ||
      lower.startsWith('::ffff:10.') ||
      lower.startsWith('::ffff:169.254.') ||
      lower.startsWith('::ffff:192.168.')
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Validates that an external webhook URL uses HTTPS, points to a public host,
 * and does not resolve to loopback, private, or cloud metadata addresses.
 */
export async function validateExternalWebhookUrl(rawUrl: string): Promise<WebhookValidationResult> {
  logFunctionCall('validateExternalWebhookUrl', { rawUrl });
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return { valid: false, error: 'Webhook URL is required.' };
  }

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { valid: false, error: 'Invalid URL format. Expected an https:// endpoint.' };
  }

  if (parsed.protocol !== 'https:') {
    return {
      valid: false,
      error: 'External Cloud MuJoCo webhooks must use https:// for secure transport.',
    };
  }

  const host = parsed.hostname.toLowerCase();
  if (
    host === 'localhost' ||
    host.endsWith('.local') ||
    host.endsWith('.internal') ||
    host === 'metadata.google.internal'
  ) {
    return {
      valid: false,
      error: 'Localhost and internal cluster hostnames are blocked for outbound webhooks.',
    };
  }

  if (net.isIP(host)) {
    if (isPrivateOrRestrictedIp(host)) {
      return { valid: false, error: 'Private, loopback, and metadata IP ranges are blocked.' };
    }
    return { valid: true, normalizedUrl: parsed.toString() };
  }

  try {
    const records = await dns.lookup(host, { all: true });
    if (!records.length) {
      return { valid: false, error: `Could not resolve hostname ${host}.` };
    }
    for (const rec of records) {
      if (isPrivateOrRestrictedIp(rec.address)) {
        return {
          valid: false,
          error: `Hostname ${host} resolves to a restricted internal IP address.`,
        };
      }
    }
  } catch {
    return { valid: false, error: `DNS lookup failed for ${host}.` };
  }

  return { valid: true, normalizedUrl: parsed.toString() };
}

/**
 * Performs a lightweight health probe against a validated Cloud MuJoCo webhook.
 */
export async function testExternalWebhook(
  url: string,
  authToken?: string | null,
): Promise<{ ok: boolean; status?: number; latencyMs: number; message: string }> {
  logFunctionCall('testExternalWebhook', { url, hasAuthToken: Boolean(authToken) });
  const validation = await validateExternalWebhookUrl(url);
  if (!validation.valid || !validation.normalizedUrl) {
    return { ok: false, latencyMs: 0, message: validation.error ?? 'Invalid webhook URL.' };
  }

  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    if (authToken && authToken.trim()) {
      headers.Authorization = `Bearer ${authToken.trim()}`;
    }
    const response = await fetch(validation.normalizedUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify({ healthcheck: true, swarm_size: APP_CONFIG.simulation.swarmSize }),
      signal: controller.signal,
    });
    const latencyMs = Date.now() - start;
    if (response.ok) {
      return {
        ok: true,
        status: response.status,
        latencyMs,
        message: `Connected (${response.status}) in ${latencyMs} ms. Ready for 20-env MuJoCo fan-out.`,
      };
    }
    return {
      ok: false,
      status: response.status,
      latencyMs,
      message: `Webhook responded with HTTP ${response.status}. Verify endpoint path and auth token.`,
    };
  } catch (err) {
    const latencyMs = Date.now() - start;
    const msg = err instanceof Error ? err.message : 'Network error reaching webhook.';
    return { ok: false, latencyMs, message: `Connection failed: ${msg}` };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Calls the external Modal / Cloud Run MuJoCo swarm worker to evaluate 20
 * environments in parallel and enriches returned telemetry with synchronized
 * 3D playback trajectories so both Cloud MP4 and 3D WebGL views remain in sync.
 */
export async function executeExternalSwarmWebhook(params: {
  webhookUrl: string;
  authToken: string | null;
  sceneSpec: SceneSpec;
  controllerCode: string;
  controllerParams: ControllerParams;
  robotModel: RobotModelId;
}): Promise<{
  rollouts: RolloutResult[];
  passed: number;
  total: number;
}> {
  logFunctionCall('executeExternalSwarmWebhook', {
    webhookUrl: params.webhookUrl,
    hasAuthToken: Boolean(params.authToken),
    robotModel: params.robotModel,
    envCount: params.sceneSpec.randomizations.length,
  });

  const validation = await validateExternalWebhookUrl(params.webhookUrl);
  if (!validation.valid || !validation.normalizedUrl) {
    throw new Error(validation.error ?? 'Invalid external Cloud MuJoCo webhook URL.');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), APP_CONFIG.upload.webhookTimeoutMs);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    if (params.authToken && params.authToken.trim()) {
      headers.Authorization = `Bearer ${params.authToken.trim()}`;
    }

    const response = await fetch(validation.normalizedUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        mjcf_xml: params.sceneSpec.mjcfXml,
        randomizations: params.sceneSpec.randomizations.map((r: DomainRandomizationParam) => ({
          id: r.id,
          friction: r.friction,
          mass: r.mass,
          xy_offset: r.xyOffset,
        })),
        controller_code: params.controllerCode,
        robot_model: params.robotModel,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const bodyText = await response.text().catch(() => '');
      throw new Error(
        `Cloud MuJoCo webhook returned HTTP ${response.status}${bodyText ? `: ${bodyText.slice(0, 180)}` : ''}`,
      );
    }

    const payload = (await response.json()) as {
      rollouts?: Array<{
        env_id?: number;
        success?: boolean;
        final_object_z?: number;
        peak_contact_force_n?: number;
        max_slip_velocity_ms?: number;
        failure_reason?: string;
        video_b64?: string;
        video_url?: string;
      }>;
      preview_videos_b64?: string[];
    };

    if (!Array.isArray(payload.rollouts) || payload.rollouts.length === 0) {
      throw new Error('Cloud MuJoCo webhook returned an empty rollouts array.');
    }

    // Generate synchronized kinematic frames so the 3D inspector mirrors webhook outcomes
    const kinematicRef = evaluateBuiltInSwarm({
      sceneSpec: params.sceneSpec,
      controllerParams: params.controllerParams,
      robotModel: params.robotModel,
    });

    const rollouts: RolloutResult[] = params.sceneSpec.randomizations.map((rand, idx) => {
      const remote = payload.rollouts?.find((r) => r.env_id === rand.id) ?? payload.rollouts?.[idx];
      const ref = kinematicRef.rollouts[idx];
      const previewB64 = remote?.video_b64 ?? payload.preview_videos_b64?.[idx] ?? null;
      const safeRemoteVideoUrl =
        typeof remote?.video_url === 'string' && remote.video_url.startsWith('https://')
          ? remote.video_url
          : null;
      const previewVideoUrl = safeRemoteVideoUrl
        ? safeRemoteVideoUrl
        : previewB64
          ? `data:video/mp4;base64,${previewB64}`
          : null;

      const finalObjectZ =
        typeof remote?.final_object_z === 'number' ? remote.final_object_z : ref.finalObjectZ;
      const peakContactForceN =
        typeof remote?.peak_contact_force_n === 'number'
          ? remote.peak_contact_force_n
          : ref.peakContactForceN;
      const maxSlipVelocityMs =
        typeof remote?.max_slip_velocity_ms === 'number'
          ? remote.max_slip_velocity_ms
          : ref.maxSlipVelocityMs;
      const success =
        typeof remote?.success === 'boolean'
          ? remote.success
          : finalObjectZ >= APP_CONFIG.simulation.targetLiftZMeters &&
            peakContactForceN <= APP_CONFIG.simulation.maxSafeContactForceN;

      return {
        envId: rand.id,
        success,
        failureReason: success
          ? null
          : (remote?.failure_reason ??
            (peakContactForceN > APP_CONFIG.simulation.maxSafeContactForceN
              ? `Contact force spike (${peakContactForceN.toFixed(1)} N > 45 N)`
              : `Insufficient lift Z (${finalObjectZ.toFixed(3)} m < 0.120 m)`)),
        finalObjectZ: Number(finalObjectZ.toFixed(4)),
        peakContactForceN: Number(peakContactForceN.toFixed(2)),
        maxSlipVelocityMs: Number(maxSlipVelocityMs.toFixed(4)),
        params: rand,
        trajectory: ref.trajectory,
        previewVideoUrl,
      };
    });

    const passed = rollouts.filter((r) => r.success).length;
    return { rollouts, passed, total: rollouts.length };
  } finally {
    clearTimeout(timer);
  }
}
