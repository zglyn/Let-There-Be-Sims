/**
 * Synchronized Dual-Axis Contact Force & Z-Lift Telemetry Chart (`TelemetryChart`)
 *
 * Renders a precision SVG dual-axis waveform chart for the currently selected
 * environment (`E01`–`E20`):
 * - Primary Series (Amber / Rose): Gripper Contact Force `F(t)` in Newtons,
 *   annotated with the `45.0 N` safety threshold dashed rule.
 * - Secondary Series (Cyan): Target Object Lift Height `Z(t)` in meters,
 *   annotated with the `0.12 m` verification threshold dashed rule.
 * - Vertical playback cursor synchronized live with the 3D Three.js stage plus
 *   interactive mouse hover crosshair snapping to exact `(t, F, Z, v_slip)`.
 *
 * Use cases:
 * - Diagnosing exact millisecond contact-force spikes or grasp slip events.
 * - Scrubbing directly along the force curve to jump the 3D robot arm to that
 *   exact kinematic keyframe.
 */

import React, { useState } from 'react';
import { APP_CONFIG } from '../config/appConfig';
import type { RolloutResult } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

interface TelemetryChartProps {
  rollout: RolloutResult | null;
  currentFrameIndex: number;
  onSeekFrame: (index: number) => void;
}

const VIEW_W = 720;
const VIEW_H = 190;
const PAD_L = 44;
const PAD_R = 44;
const PAD_T = 18;
const PAD_B = 26;

/**
 * Renders the interactive dual-axis SVG telemetry chart synced with 3D replay.
 */
export const TelemetryChart: React.FC<TelemetryChartProps> = ({
  rollout,
  currentFrameIndex,
  onSeekFrame,
}) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const frames = rollout?.trajectory ?? [];
  const plotW = VIEW_W - PAD_L - PAD_R;
  const plotH = VIEW_H - PAD_T - PAD_B;

  const maxForceAxis = Math.max(
    60,
    Math.ceil((rollout?.peakContactForceN ?? 50) / 10) * 10 + 10,
  );
  const maxZAxis = 0.28;

  const xForIndex = (idx: number): number =>
    frames.length <= 1 ? PAD_L : PAD_L + (idx / (frames.length - 1)) * plotW;
  const yForForce = (force: number): number =>
    PAD_T + plotH - Math.min(1, Math.max(0, force / maxForceAxis)) * plotH;
  const yForZ = (z: number): number =>
    PAD_T + plotH - Math.min(1, Math.max(0, z / maxZAxis)) * plotH;

  const forcePath = frames
    .map((f, i) => `${i === 0 ? 'M' : 'L'}${xForIndex(i).toFixed(1)},${yForForce(f.contactForceN).toFixed(1)}`)
    .join(' ');
  const zPath = frames
    .map((f, i) => `${i === 0 ? 'M' : 'L'}${xForIndex(i).toFixed(1)},${yForZ(f.objectPos.y).toFixed(1)}`)
    .join(' ');

  const forceLimitY = yForForce(APP_CONFIG.simulation.maxSafeContactForceN);
  const zTargetY = yForZ(APP_CONFIG.simulation.targetLiftZMeters);

  const activeIdx = hoverIndex !== null ? hoverIndex : currentFrameIndex;
  const activeFrame = frames[activeIdx] ?? null;

  /**
   * Computes the nearest trajectory frame index from an SVG pointer event.
   */
  const indexFromClientX = (svgEl: SVGSVGElement, clientX: number): number => {
    const rect = svgEl.getBoundingClientRect();
    const relX = ((clientX - rect.left) / rect.width) * VIEW_W;
    const norm = Math.max(0, Math.min(1, (relX - PAD_L) / plotW));
    return Math.round(norm * Math.max(0, frames.length - 1));
  };

  return (
    <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <h3 className="font-display text-sm font-semibold text-slate-100">
            Contact Force & Lift Telemetry
          </h3>
          <div className="flex items-center gap-3 text-[11px] font-mono-tabular text-slate-400">
            <span className="text-amber-400">— Force (N)</span>
            <span className="text-cyan-400">— Object Z (m)</span>
          </div>
        </div>

        {activeFrame ? (
          <div className="flex items-center gap-3 text-xs font-mono-tabular">
            <span className="text-slate-400">
              t={activeFrame.timeSec.toFixed(1)}s /{' '}
              {(
                frames[frames.length - 1]?.timeSec ??
                rollout?.durationSec ??
                3
              ).toFixed(1)}
              s
            </span>
            <span
              className={
                activeFrame.contactForceN > APP_CONFIG.simulation.maxSafeContactForceN
                  ? 'text-rose-400 font-semibold'
                  : 'text-amber-400'
              }
            >
              {activeFrame.contactForceN.toFixed(1)} N
            </span>
            <span className="text-cyan-400">{activeFrame.objectPos.y.toFixed(3)} m</span>
            <span className="text-slate-400">
              slip {activeFrame.slipVelocityMs.toFixed(3)} m/s
            </span>
          </div>
        ) : (
          <span className="text-xs font-mono-tabular text-slate-500">
            Click chart to scrub 3D trajectory
          </span>
        )}
      </div>

      <div className="w-full overflow-hidden">
        <svg
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          className="w-full h-44 cursor-crosshair select-none"
          onMouseMove={(e) => {
            if (!frames.length) return;
            setHoverIndex(indexFromClientX(e.currentTarget, e.clientX));
          }}
          onMouseLeave={() => setHoverIndex(null)}
          onClick={(e) => {
            if (!frames.length) return;
            const idx = indexFromClientX(e.currentTarget, e.clientX);
            logFunctionCall('TelemetryChart.seekFrame', { frameIndex: idx });
            onSeekFrame(idx);
          }}
        >
          {/* Horizontal hairline grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = PAD_T + t * plotH;
            const forceVal = Math.round(maxForceAxis * (1 - t));
            const zVal = (maxZAxis * (1 - t)).toFixed(2);
            return (
              <g key={t}>
                <line
                  x1={PAD_L}
                  y1={y}
                  x2={VIEW_W - PAD_R}
                  y2={y}
                  stroke="#1E293B"
                  strokeWidth={1}
                />
                <text
                  x={PAD_L - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-500 text-[9px] font-mono"
                >
                  {forceVal}N
                </text>
                <text
                  x={VIEW_W - PAD_R + 6}
                  y={y + 3}
                  textAnchor="start"
                  className="fill-slate-500 text-[9px] font-mono"
                >
                  {zVal}m
                </text>
              </g>
            );
          })}

          {/* 45.0 N Safety Force Limit Reference Line */}
          <line
            x1={PAD_L}
            y1={forceLimitY}
            x2={VIEW_W - PAD_R}
            y2={forceLimitY}
            stroke="#F43F5E"
            strokeWidth={1}
            strokeDasharray="4 4"
            opacity={0.75}
          />
          <text
            x={PAD_L + 6}
            y={forceLimitY - 4}
            className="fill-rose-400 text-[9px] font-mono"
          >
            Limit 45.0 N
          </text>

          {/* 0.12 m Target Lift Reference Line */}
          <line
            x1={PAD_L}
            y1={zTargetY}
            x2={VIEW_W - PAD_R}
            y2={zTargetY}
            stroke="#06B6D4"
            strokeWidth={1}
            strokeDasharray="3 4"
            opacity={0.6}
          />
          <text
            x={VIEW_W - PAD_R - 6}
            y={zTargetY - 4}
            textAnchor="end"
            className="fill-cyan-400 text-[9px] font-mono"
          >
            Target 0.12 m
          </text>

          {/* Waveform Curves */}
          {frames.length > 1 ? (
            <>
              <path
                d={zPath}
                fill="none"
                stroke="#06B6D4"
                strokeWidth={2}
                strokeLinejoin="round"
              />
              <path
                d={forcePath}
                fill="none"
                stroke={
                  (rollout?.peakContactForceN ?? 0) > APP_CONFIG.simulation.maxSafeContactForceN
                    ? '#F43F5E'
                    : '#F59E0B'
                }
                strokeWidth={2}
                strokeLinejoin="round"
              />

              {/* Active Playback / Hover Crosshair */}
              {activeFrame && (
                <g>
                  <line
                    x1={xForIndex(activeIdx)}
                    y1={PAD_T}
                    x2={xForIndex(activeIdx)}
                    y2={PAD_T + plotH}
                    stroke="#94A3B8"
                    strokeWidth={1}
                    strokeDasharray="2 2"
                  />
                  <circle
                    cx={xForIndex(activeIdx)}
                    cy={yForForce(activeFrame.contactForceN)}
                    r={3.5}
                    fill="#F59E0B"
                  />
                  <circle
                    cx={xForIndex(activeIdx)}
                    cy={yForZ(activeFrame.objectPos.y)}
                    r={3.5}
                    fill="#06B6D4"
                  />
                </g>
              )}
            </>
          ) : (
            <text
              x={VIEW_W / 2}
              y={VIEW_H / 2}
              textAnchor="middle"
              className="fill-slate-500 text-xs font-mono"
            >
              Run pipeline to stream 500 Hz contact-force and Z-lift telemetry
            </text>
          )}

          {/* X-axis timestamps scaled dynamically to full rollout duration */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, idx) => {
            const totalDurationSec =
              frames[frames.length - 1]?.timeSec ??
              rollout?.durationSec ??
              APP_CONFIG.simulation.defaultRolloutDurationSec;
            const sec = frac * totalDurationSec;
            const x = PAD_L + (idx / 4) * plotW;
            const label =
              totalDurationSec >= 60
                ? `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, '0')}`
                : totalDurationSec >= 10
                  ? `${sec.toFixed(1)}s`
                  : `${sec.toFixed(2)}s`;
            return (
              <text
                key={idx}
                x={x}
                y={VIEW_H - 6}
                textAnchor="middle"
                className="fill-slate-500 text-[9px] font-mono"
              >
                {label}
              </text>
            );
          })}
        </svg>
      </div>
    </section>
  );
};
