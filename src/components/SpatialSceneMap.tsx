/**
 * Interactive 2D Metric Spatial Scene Map (`SpatialSceneMap`)
 *
 * Renders a calibrated top-down metric occupancy & workspace map reconstructed
 * from uploaded video footage by Gemini 4 Carbon. Designed primarily for
 * robot-free captures (`scene_map_only` mode) where the video contains a desk,
 * tabletop, or workspace without a robotic arm, while remaining inspectable
 * during full robot policy repair runs as well.
 *
 * Features:
 * - Metric X/Z coordinate grid (in centimeters) with tabletop boundary frame.
 * - Footprints for the primary target object and all mapped obstacles (circle
 *   or oriented box scaled to real metric dimensions) with clearance rings.
 * - Spatial zones (`robot_mount`, `clear_zone`, `hazard_zone`) and the
 *   selected robot's reachable workspace envelope arc.
 * - Click-to-inspect object properties (3D pose, size, mass, friction, material).
 * - One-click `Export MJCF Scene (.xml)` and optional `Mount Robot & Simulate
 *   Policy` trigger to spawn SO-ARM101 or Franka Panda onto the mapped scene.
 *
 * Use cases:
 * - Reviewing a simulation-ready real-to-sim map from a smartphone scan of any
 *   tabletop or room without needing a physical robot on camera.
 * - Deciding where to mount a robot base before launching a 20-env policy run.
 */

import React, { useMemo, useState } from 'react';
import { Download, MapPin, Play } from 'lucide-react';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type { SceneObjectSpec, SceneSpec } from '../types/robotics';
import { downloadTextArtifact } from '../utils/fileDownload';
import { logFunctionCall } from '../utils/logger';

interface SpatialSceneMapProps {
  sessionId: string;
  sceneSpec: SceneSpec | null;
  robotModel: RobotModelId;
  isPipelineRunning: boolean;
  onMountAndSimulate: () => Promise<void>;
  onExported: (message: string) => void;
}

const SVG_W = 680;
const SVG_H = 380;
const PAD = 38;

/**
 * Renders the interactive top-down 2D metric spatial map and object inspector.
 */
export const SpatialSceneMap: React.FC<SpatialSceneMapProps> = ({
  sessionId,
  sceneSpec,
  robotModel,
  isPipelineRunning,
  onMountAndSimulate,
  onExported,
}) => {
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const allObjects = useMemo<Array<{ spec: SceneObjectSpec; isTarget: boolean }>>(() => {
    if (!sceneSpec) return [];
    return [
      { spec: sceneSpec.targetObject, isTarget: true },
      ...sceneSpec.obstacles.map((o) => ({ spec: o, isTarget: false })),
    ];
  }, [sceneSpec]);

  const selectedEntry =
    allObjects.find((o) => o.spec.name === selectedName) ?? allObjects[0] ?? null;

  if (!sceneSpec) {
    return (
      <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-8 text-center">
        <h3 className="font-display text-base font-semibold text-slate-100">
          2D Metric Spatial Map
        </h3>
        <p className="mt-1.5 text-xs text-slate-400 max-w-md mx-auto">
          Upload any tabletop or workspace video (with or without a robotic arm) and run Gemini 4 Carbon to generate a calibrated 2D occupancy map and simulation-ready MuJoCo scene.
        </p>
      </section>
    );
  }

  // World coordinate bounds centered around tabletop [x: -0.26..0.56, z: -0.34..0.34]
  const tableX = sceneSpec.tableDimensions.x;
  const tableZ = sceneSpec.tableDimensions.z;
  const minX = -0.24;
  const maxX = Math.max(0.56, tableX - 0.16);
  const halfZ = Math.max(0.36, tableZ * 0.56);
  const minZ = -halfZ;
  const maxZ = halfZ;

  const spanX = maxX - minX;
  const spanZ = maxZ - minZ;
  const scale = Math.min((SVG_W - PAD * 2) / spanZ, (SVG_H - PAD * 2) / spanX);
  const originScreenX = SVG_W / 2;
  const originScreenY = SVG_H - PAD - ((0 - minX) / spanX) * (SVG_H - PAD * 2);

  // Map Three.js metric coordinates (X forward = up on map, Z lateral = right on map)
  const toMapX = (z: number): number => originScreenX + z * scale;
  const toMapY = (x: number): number => originScreenY - x * scale;

  const robotReachPx = APP_CONFIG.robots[robotModel].reachMeters * scale;
  const baseX = toMapX(sceneSpec.recommendedRobotBase.z);
  const baseY = toMapY(sceneSpec.recommendedRobotBase.x);

  const isMapOnly = sceneSpec.sceneMode === 'scene_map_only';

  return (
    <section className="bg-[#111622] border border-[#1E293B] rounded-lg p-4 flex flex-col gap-4">
      {/* Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1E293B]">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2.5">
            <h3 className="font-display text-base font-semibold text-slate-100">
              Calibrated 2D Spatial Scene Map
            </h3>
            <span className="font-mono-tabular text-xs text-cyan-400">
              {isMapOnly ? 'Scene Map Mode · Simulation-Ready' : 'Robot Workcell Map'}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            {sceneSpec.detectedRobotDescription} · {allObjects.length} mapped object
            {allObjects.length === 1 ? '' : 's'} · Workspace {(tableX * 100).toFixed(0)} ×{' '}
            {(tableZ * 100).toFixed(0)} cm
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => {
              logFunctionCall('SpatialSceneMap.exportMjcf', { sessionId });
              downloadTextArtifact(
                `${sessionId}_scene_map.mjcf.xml`,
                sceneSpec.mjcfXml,
                'application/xml;charset=utf-8',
              );
              onExported(`Exported ${sessionId}_scene_map.mjcf.xml`);
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-[#161D2C] border border-[#2D3B52] rounded-md hover:border-cyan-400/60 hover:text-cyan-200 transition-colors whitespace-nowrap cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export MJCF (.xml)</span>
          </button>

          {isMapOnly && (
            <button
              type="button"
              disabled={isPipelineRunning}
              onClick={() => {
                logFunctionCall('SpatialSceneMap.mountAndSimulate', {
                  sessionId,
                  robotModel,
                });
                void onMountAndSimulate();
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-amber-500 rounded-md hover:bg-amber-400 disabled:opacity-50 transition-colors whitespace-nowrap cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>
                {isPipelineRunning
                  ? 'Running 20-Env Swarm...'
                  : `Mount ${robotModel === 'so_arm101' ? 'SO-ARM101' : 'Franka'} & Simulate`}
              </span>
            </button>
          )}
        </div>
      </div>

      {/* Map Canvas + Right Object Inspector */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-stretch">
        <div className="xl:col-span-8 bg-[#07090E] border border-[#1E293B] rounded-md overflow-hidden">
          <svg
            viewBox={`0 0 ${SVG_W} ${SVG_H}`}
            className="w-full h-[320px] select-none"
            role="img"
            aria-label="Top-down metric spatial map of reconstructed simulation scene"
          >
            {/* Metric 10cm Grid Lines */}
            {[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3].map((zTick) => {
              const xPos = toMapX(zTick);
              return (
                <g key={`z_${zTick}`}>
                  <line
                    x1={xPos}
                    y1={PAD / 2}
                    x2={xPos}
                    y2={SVG_H - PAD / 2}
                    stroke={zTick === 0 ? '#2D3B52' : '#141B29'}
                    strokeWidth={zTick === 0 ? 1.2 : 1}
                  />
                  <text
                    x={xPos}
                    y={SVG_H - 8}
                    textAnchor="middle"
                    className="fill-slate-500 text-[9px] font-mono"
                  >
                    {Math.round(zTick * 100)}cm
                  </text>
                </g>
              );
            })}

            {[-0.1, 0, 0.1, 0.2, 0.3, 0.4].map((xTick) => {
              const yPos = toMapY(xTick);
              if (yPos < 16 || yPos > SVG_H - 20) return null;
              return (
                <g key={`x_${xTick}`}>
                  <line
                    x1={PAD / 2}
                    y1={yPos}
                    x2={SVG_W - PAD / 2}
                    y2={yPos}
                    stroke={xTick === 0 ? '#2D3B52' : '#141B29'}
                    strokeWidth={xTick === 0 ? 1.2 : 1}
                  />
                  <text
                    x={12}
                    y={yPos + 3}
                    textAnchor="start"
                    className="fill-slate-500 text-[9px] font-mono"
                  >
                    {Math.round(xTick * 100)}cm
                  </text>
                </g>
              );
            })}

            {/* Calibrated Tabletop Boundary */}
            <rect
              x={toMapX(-tableZ / 2)}
              y={toMapY(0.14 + tableX / 2)}
              width={tableZ * scale}
              height={tableX * scale}
              fill="rgba(17, 22, 34, 0.45)"
              stroke="#334155"
              strokeWidth={1.5}
              strokeDasharray="6 4"
              rx={4}
            />

            {/* Reachable Robot Workspace Arc */}
            <circle
              cx={baseX}
              cy={baseY}
              r={robotReachPx}
              fill="none"
              stroke="rgba(6, 182, 212, 0.25)"
              strokeWidth={1.2}
              strokeDasharray="4 4"
            />

            {/* Spatial Zones (clear, hazard, robot_mount) */}
            {sceneSpec.spatialZones.map((zone, idx) => {
              const zx = toMapX(zone.centerZ);
              const zy = toMapY(zone.centerX);
              const zr = Math.max(14, zone.radiusM * scale);
              const stroke =
                zone.kind === 'hazard_zone'
                  ? '#F43F5E'
                  : zone.kind === 'robot_mount'
                    ? '#F59E0B'
                    : '#10B981';
              const fill =
                zone.kind === 'hazard_zone'
                  ? 'rgba(244, 63, 94, 0.08)'
                  : zone.kind === 'robot_mount'
                    ? 'rgba(245, 158, 11, 0.1)'
                    : 'rgba(16, 185, 129, 0.07)';
              return (
                <g key={`${zone.label}_${idx}`}>
                  <circle
                    cx={zx}
                    cy={zy}
                    r={zr}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={1.2}
                    strokeDasharray="4 3"
                  />
                  <text
                    x={zx}
                    y={zy + zr + 11}
                    textAnchor="middle"
                    className="fill-slate-400 text-[9px] font-mono"
                  >
                    {zone.label}
                  </text>
                </g>
              );
            })}

            {/* Recommended Robot Base Anchor Marker */}
            <g>
              <circle
                cx={baseX}
                cy={baseY}
                r={7}
                fill="#0B0E17"
                stroke="#F59E0B"
                strokeWidth={2}
              />
              <line x1={baseX - 11} y1={baseY} x2={baseX + 11} y2={baseY} stroke="#F59E0B" strokeWidth={1} />
              <line x1={baseX} y1={baseY - 11} x2={baseX} y2={baseY + 11} stroke="#F59E0B" strokeWidth={1} />
            </g>

            {/* Reconstructed Scene Objects & Obstacles */}
            {allObjects.map(({ spec, isTarget }) => {
              const ox = toMapX(spec.position.z);
              const oy = toMapY(spec.position.x);
              const wPx = Math.max(18, spec.dimensions.z * scale);
              const hPx = Math.max(18, spec.dimensions.x * scale);
              const rPx = Math.max(11, (Math.max(spec.dimensions.x, spec.dimensions.z) * 0.5) * scale);
              const isSelected = selectedEntry?.spec.name === spec.name;

              return (
                <g
                  key={spec.name}
                  className="cursor-pointer"
                  onClick={() => {
                    logFunctionCall('SpatialSceneMap.selectObject', { name: spec.name });
                    setSelectedName(spec.name);
                  }}
                >
                  {/* Safety Clearance Halo */}
                  <circle
                    cx={ox}
                    cy={oy}
                    r={Math.max(rPx, Math.max(wPx, hPx) * 0.5) + 12}
                    fill="none"
                    stroke={isTarget ? 'rgba(6, 182, 212, 0.45)' : 'rgba(148, 163, 184, 0.25)'}
                    strokeWidth={isSelected ? 1.8 : 1}
                    strokeDasharray="3 3"
                  />

                  {spec.shape === 'box' ? (
                    <rect
                      x={ox - wPx / 2}
                      y={oy - hPx / 2}
                      width={wPx}
                      height={hPx}
                      rx={3}
                      fill={isTarget ? 'rgba(6, 182, 212, 0.28)' : 'rgba(100, 116, 139, 0.35)'}
                      stroke={isSelected ? '#F59E0B' : isTarget ? '#06B6D4' : '#94A3B8'}
                      strokeWidth={isSelected ? 2.2 : 1.5}
                    />
                  ) : (
                    <circle
                      cx={ox}
                      cy={oy}
                      r={rPx}
                      fill={isTarget ? 'rgba(6, 182, 212, 0.28)' : 'rgba(100, 116, 139, 0.35)'}
                      stroke={isSelected ? '#F59E0B' : isTarget ? '#06B6D4' : '#94A3B8'}
                      strokeWidth={isSelected ? 2.2 : 1.5}
                    />
                  )}

                  <text
                    x={ox}
                    y={oy - Math.max(rPx, hPx / 2) - 6}
                    textAnchor="middle"
                    className={`text-[10px] font-mono ${
                      isTarget ? 'fill-cyan-300 font-semibold' : 'fill-slate-200'
                    }`}
                  >
                    {spec.name}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Right Selected Object & Calibration Inspector */}
        <div className="xl:col-span-4 bg-[#0B0E17] border border-[#1E293B] rounded-md p-4 flex flex-col justify-between gap-4">
          {selectedEntry && (
            <div className="flex flex-col gap-3">
              <div className="flex items-start justify-between gap-2 pb-2.5 border-b border-[#1E293B]">
                <div>
                  <span className="font-mono-tabular text-[10px] text-cyan-400">
                    {selectedEntry.isTarget ? 'Primary Target Object' : 'Mapped Obstacle / Asset'}
                  </span>
                  <h4 className="font-display text-sm font-semibold text-slate-100 mt-0.5">
                    {selectedEntry.spec.name}
                  </h4>
                </div>
                <MapPin className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              </div>

              <dl className="grid grid-cols-2 gap-x-3 gap-y-2.5 text-xs">
                <div>
                  <dt className="text-slate-500">Pose (X, Y, Z)</dt>
                  <dd className="font-mono-tabular text-slate-200 mt-0.5">
                    {selectedEntry.spec.position.x.toFixed(2)}, {selectedEntry.spec.position.y.toFixed(2)},{' '}
                    {selectedEntry.spec.position.z.toFixed(2)} m
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Dimensions</dt>
                  <dd className="font-mono-tabular text-slate-200 mt-0.5">
                    {(selectedEntry.spec.dimensions.x * 100).toFixed(1)}×
                    {(selectedEntry.spec.dimensions.y * 100).toFixed(1)}×
                    {(selectedEntry.spec.dimensions.z * 100).toFixed(1)} cm
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Estimated Mass</dt>
                  <dd className="font-mono-tabular text-slate-200 mt-0.5">
                    {selectedEntry.spec.massKg.toFixed(2)} kg
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Surface Friction μ</dt>
                  <dd className="font-mono-tabular text-slate-200 mt-0.5">
                    {selectedEntry.spec.friction.toFixed(2)}
                  </dd>
                </div>
                <div className="col-span-2">
                  <dt className="text-slate-500">Material & Primitive</dt>
                  <dd className="text-slate-300 mt-0.5">
                    {selectedEntry.spec.materialLabel} · {selectedEntry.spec.shape}
                  </dd>
                </div>
              </dl>
            </div>
          )}

          {/* Map Legend & Calibration Notes */}
          <div className="pt-3 border-t border-[#1E293B] flex flex-col gap-2 text-[11px] text-slate-400">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono-tabular">
              <span className="text-cyan-400">○ Target</span>
              <span className="text-slate-300">○ Obstacle</span>
              <span className="text-amber-400">⊕ Robot Mount</span>
              <span className="text-rose-400">◌ Hazard</span>
            </div>
            <p className="leading-relaxed">{sceneSpec.calibrationNotes}</p>
          </div>
        </div>
      </div>
    </section>
  );
};
