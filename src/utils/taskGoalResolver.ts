/**
 * Shared Task Goal Placement Resolver (`taskGoalResolver.ts`)
 *
 * Computes a deterministic, obstacle-free tabletop destination coordinate
 * ("Goal Placement Pad") for any reconstructed `SceneSpec`. Sharing this single
 * resolver across the backend 20-environment physics evaluator, the browser's
 * live 3D physics sandbox, and the Three.js Goal Pad mesh guarantees that every
 * pick-lift-transport-place task executes to the exact same physical landing
 * zone visible on the workbench.
 *
 * Use cases:
 * - Resolving a clean drop-off coordinate from Gemini 4 Carbon's `spatialZones`
 *   (`clear_zone`) when available.
 * - Falling back to an obstacle-free opposite-side tabletop landing pad within
 *   safe robot reach whenever no explicit clear zone is provided.
 */

import type { SceneSpec } from '../types/robotics';
import { logFunctionCall } from './logger';

export interface TaskGoalPose {
  x: number;
  z: number;
  radiusM: number;
  label: string;
}

/**
 * Clamps a numeric scalar to the inclusive [min, max] range.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Measures the minimum horizontal clearance between candidate coordinate (x, z)
 * and every obstacle plus the target object's pickup origin.
 */
function computeCandidateClearance(
  x: number,
  z: number,
  sceneSpec: SceneSpec,
  pickupX: number,
  pickupZ: number,
): number {
  const targetRadius =
    Math.max(sceneSpec.targetObject.dimensions.x, sceneSpec.targetObject.dimensions.z) * 0.5;
  const distFromPickup = Math.hypot(x - pickupX, z - pickupZ);
  let minClearance = distFromPickup - targetRadius * 2.2;

  for (const obs of sceneSpec.obstacles) {
    const obsRadius = Math.max(obs.dimensions.x, obs.dimensions.z) * 0.5;
    const distObs = Math.hypot(x - obs.position.x, z - obs.position.z);
    const gap = distObs - obsRadius - targetRadius - 0.015;
    if (gap < minClearance) {
      minClearance = gap;
    }
  }
  return minClearance;
}

/**
 * Resolves a physically reachable, obstacle-free Goal Placement Pad coordinate
 * on the tabletop for completing pick-and-place manipulation tasks.
 */
export function resolveTaskGoalPose(sceneSpec: SceneSpec): TaskGoalPose {
  const pickupX = sceneSpec.targetObject.position.x;
  const pickupZ = sceneSpec.targetObject.position.z;
  const targetRadius = Math.max(
    0.032,
    Math.max(sceneSpec.targetObject.dimensions.x, sceneSpec.targetObject.dimensions.z) * 0.5,
  );
  const padRadius = clamp(targetRadius + 0.022, 0.048, 0.075);

  // 0. If Carbon retargeted a human hand demonstration, honor the demonstrated goal drop position
  const handPlace = sceneSpec.handDemoRetarget?.detected
    ? sceneSpec.handDemoRetarget.placePos
    : null;
  if (handPlace && Number.isFinite(handPlace.x) && Number.isFinite(handPlace.z)) {
    const hx = clamp(handPlace.x, 0.13, 0.34);
    const hz = clamp(handPlace.z, -0.24, 0.24);
    const reach = Math.hypot(hx, hz);
    const sepFromPickup = Math.hypot(hx - pickupX, hz - pickupZ);
    if (reach <= 0.38 && sepFromPickup >= 0.08) {
      const resolved: TaskGoalPose = {
        x: Number(hx.toFixed(4)),
        z: Number(hz.toFixed(4)),
        radiusM: padRadius,
        label: 'Hand Demo Goal Pad',
      };
      logFunctionCall('resolveTaskGoalPose', { source: 'hand_demo_retarget', resolved });
      return resolved;
    }
  }

  // 1. Prefer an explicit `clear_zone` from Gemini 4 Carbon if within arm reach & far enough from pickup
  const clearZone = sceneSpec.spatialZones?.find((zone) => zone.kind === 'clear_zone');
  if (clearZone) {
    const cx = clamp(clearZone.centerX, 0.14, 0.31);
    const cz = clamp(clearZone.centerZ, -0.22, 0.22);
    const reach = Math.hypot(cx, cz);
    const sepFromPickup = Math.hypot(cx - pickupX, cz - pickupZ);
    const clearance = computeCandidateClearance(cx, cz, sceneSpec, pickupX, pickupZ);
    if (reach <= 0.36 && sepFromPickup >= 0.11 && clearance >= 0.01) {
      const resolved: TaskGoalPose = {
        x: Number(cx.toFixed(4)),
        z: Number(cz.toFixed(4)),
        radiusM: padRadius,
        label: clearZone.label || 'Goal Drop Zone',
      };
      logFunctionCall('resolveTaskGoalPose', { source: 'clear_zone', resolved });
      return resolved;
    }
  }

  // 2. Search reachable opposite-side candidate coordinates and pick the most obstacle-free landing spot
  const preferredSign = pickupZ >= 0 ? -1 : 1;
  const candidates: Array<{ x: number; z: number }> = [
    { x: clamp(pickupX, 0.16, 0.28), z: preferredSign * 0.16 },
    { x: 0.22, z: preferredSign * 0.19 },
    { x: 0.26, z: preferredSign * 0.14 },
    { x: 0.18, z: preferredSign * 0.14 },
    { x: 0.24, z: -preferredSign * 0.18 },
    { x: 0.17, z: 0.0 },
  ];

  let best = candidates[0];
  let bestScore = -Infinity;

  for (const cand of candidates) {
    const reach = Math.hypot(cand.x, cand.z);
    if (reach > 0.36) continue;
    const sep = Math.hypot(cand.x - pickupX, cand.z - pickupZ);
    const clearance = computeCandidateClearance(cand.x, cand.z, sceneSpec, pickupX, pickupZ);
    const score = clearance * 2.0 + Math.min(sep, 0.24) * 0.6;
    if (score > bestScore) {
      bestScore = score;
      best = cand;
    }
  }

  const resolved: TaskGoalPose = {
    x: Number(best.x.toFixed(4)),
    z: Number(best.z.toFixed(4)),
    radiusM: padRadius,
    label: 'Goal Drop Zone',
  };
  logFunctionCall('resolveTaskGoalPose', { source: 'computed_open_zone', resolved });
  return resolved;
}
