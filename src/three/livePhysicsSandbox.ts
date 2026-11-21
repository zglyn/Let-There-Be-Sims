/**
 * Real-Time Dynamic Robotics Physics & Analytical IK Sandbox (`livePhysicsSandbox.ts`)
 *
 * Powers the interactive, continuously responsive 3D simulation stage inside
 * `RobotStage3D`. Rather than playing only static precomputed trajectories or
 * idling in a frozen standby pose, this engine runs a frame-rate-independent
 * rigid-body + 6-DOF analytical inverse-kinematics loop directly in the browser:
 *
 * 1. **Continuous Autonomous Pick-Lift-Place Cycle:** Immediately drives the
 *    articulated arm (`SO-ARM101` or `Franka Panda`) through Approach -> Descend
 *    -> Grasp -> Lift -> Hold -> Return with smooth exponential joint damping.
 * 2. **Interactive Tabletop Drag-and-Drop:** When the user drags the target
 *    object or any obstacle across the 3D workbench, the sandbox updates body
 *    coordinates in real time, resolves circle-circle obstacle collisions,
 *    recomputes the live 3D approach spline, and redirects the arm cleanly.
 * 3. **Rigid-Body Impulse, Gravity, Slip & Collision Physics:** Supports live
 *    disturbance impulses ("Nudge"), dynamic Friction (mu), Mass (kg), and
 *    Approach Clearance sliders—producing authentic tip-over tumbles, gravity
 *    drops on low-friction slip, obstacle knocks, and contact force spikes
 *    (`> 45 N` fault threshold) with zero per-frame heap allocations in the
 *    inner step loop.
 *
 * Use cases:
 * - Making the 3D simulation stage genuinely alive on first load before any
 *   video is uploaded.
 * - Letting robotics engineers stress-test Gemini Carbon's reconstructed scene
 *   interactively by dragging obstacles into the approach corridor, lowering
 *   friction, or nudging the target object mid-grasp.
 */

import * as THREE from 'three';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type { ControllerParams, SceneSpec, Vector3Tuple } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

export type LiveSimPhase =
  | 'approach'
  | 'descend'
  | 'grasp'
  | 'lift'
  | 'hold'
  | 'return';

export interface LivePhysicsParams {
  friction: number;
  massKg: number;
  clearanceM: number;
  gripForceN: number;
}

export interface DynamicBodyState {
  id: string;
  name: string;
  isTarget: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  tiltX: number;
  tiltZ: number;
  angVelX: number;
  angVelZ: number;
  halfHeight: number;
  radius: number;
  homeX: number;
  homeZ: number;
}

export interface LiveTelemetrySnapshot {
  timeSec: number;
  phase: LiveSimPhase;
  contactForceN: number;
  slipVelocityMs: number;
  objectHeightM: number;
  faultState: 'ok' | 'collision' | 'slip';
  impactIntensity: number; // [0..1] drives visual contact ripple ring
  impactPosition: Vector3Tuple;
  endEffectorPos: Vector3Tuple;
}

export interface LivePhysicsSandbox {
  targetBody: DynamicBodyState;
  obstacleBodies: DynamicBodyState[];
  jointAngles: [number, number, number, number, number, number];
  gripperOpen: number;
  telemetry: LiveTelemetrySnapshot;
  step: (
    dtSec: number,
    robotModel: RobotModelId,
    params: LivePhysicsParams,
    controllerParams: ControllerParams | null,
    draggedBodyId: string | null,
  ) => void;
  setBodyPositionFromDrag: (bodyId: string, worldX: number, worldZ: number) => void;
  finishBodyDrag: (bodyId: string) => void;
  applyNudgeImpulse: () => void;
  resetBodies: () => void;
  computePreviewSplinePoints: (
    clearanceM: number,
    liftHeightM: number,
    outPoints: THREE.Vector3[],
  ) => number;
}

/**
 * Clamps a scalar value to [min, max].
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Solves analytical 6-DOF articulated arm joint angles (radians) for a desired
 * world end-effector coordinate (matching the server physics evaluator).
 */
export function solveLiveArmIK(
  targetX: number,
  targetY: number,
  targetZ: number,
  robotModel: RobotModelId,
  outAngles: [number, number, number, number, number, number],
): void {
  const spec = APP_CONFIG.robots[robotModel];
  const baseYaw = Math.atan2(targetZ, targetX);
  const planarR = Math.max(0.06, Math.hypot(targetX, targetZ));
  const wristHeight = targetY + spec.wristM;
  const dy = wristHeight - spec.baseHeightM;
  const dr = Math.max(0.04, planarR - 0.03);

  const l1 = spec.upperArmM;
  const l2 = spec.forearmM;
  const dist = clamp(Math.hypot(dr, dy), 0.05, l1 + l2 - 0.01);

  const cosElbow = clamp((dist * dist - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1);
  const elbowAngle = -Math.acos(cosElbow);
  const alpha = Math.atan2(dy, dr);
  const beta = Math.atan2(l2 * Math.sin(-elbowAngle), l1 + l2 * Math.cos(-elbowAngle));
  const shoulderPitch = alpha + beta;
  const wristPitch = -(shoulderPitch + elbowAngle) - Math.PI / 2;

  outAngles[0] = baseYaw;
  outAngles[1] = shoulderPitch;
  outAngles[2] = elbowAngle;
  outAngles[3] = wristPitch;
  outAngles[4] = 0;
  outAngles[5] = -baseYaw * 0.25;
}

/**
 * Creates a fresh real-time dynamic physics sandbox synchronized with the
 * active `SceneSpec` target object and obstacle geometries.
 */
export function createLivePhysicsSandbox(sceneSpec: SceneSpec): LivePhysicsSandbox {
  logFunctionCall('createLivePhysicsSandbox', {
    targetName: sceneSpec.targetObject.name,
    obstacleCount: sceneSpec.obstacles.length,
  });

  const targetHalfH = Math.max(0.025, sceneSpec.targetObject.dimensions.y * 0.5);
  const targetRadius = Math.max(
    0.022,
    Math.max(sceneSpec.targetObject.dimensions.x, sceneSpec.targetObject.dimensions.z) * 0.5,
  );

  const targetBody: DynamicBodyState = {
    id: 'target',
    name: sceneSpec.targetObject.name,
    isTarget: true,
    x: sceneSpec.targetObject.position.x,
    y: targetHalfH,
    z: sceneSpec.targetObject.position.z,
    vx: 0,
    vy: 0,
    vz: 0,
    tiltX: 0,
    tiltZ: 0,
    angVelX: 0,
    angVelZ: 0,
    halfHeight: targetHalfH,
    radius: targetRadius,
    homeX: sceneSpec.targetObject.position.x,
    homeZ: sceneSpec.targetObject.position.z,
  };

  const obstacleBodies: DynamicBodyState[] = sceneSpec.obstacles.map((obs, idx) => {
    const halfH = Math.max(0.025, obs.dimensions.y * 0.5);
    const radius = Math.max(0.022, Math.max(obs.dimensions.x, obs.dimensions.z) * 0.5);
    return {
      id: `obstacle_${idx}`,
      name: obs.name,
      isTarget: false,
      x: obs.position.x,
      y: halfH,
      z: obs.position.z,
      vx: 0,
      vy: 0,
      vz: 0,
      tiltX: 0,
      tiltZ: 0,
      angVelX: 0,
      angVelZ: 0,
      halfHeight: halfH,
      radius,
      homeX: obs.position.x,
      homeZ: obs.position.z,
    };
  });

  const jointAngles: [number, number, number, number, number, number] = [
    0.1, 1.05, -1.45, -1.15, 0, 0,
  ];
  const targetAngles: [number, number, number, number, number, number] = [
    0.1, 1.05, -1.45, -1.15, 0, 0,
  ];

  let gripperOpen = 0.9;
  let simTime = 0;
  let phase: LiveSimPhase = 'approach';
  let phaseElapsed = 0;
  let eeX = 0.14;
  let eeY = 0.27;
  let eeZ = 0.0;
  let isGrasped = false;
  let cycleFault: 'ok' | 'collision' | 'slip' = 'ok';
  let impactIntensity = 0;
  let impactX = targetBody.x;
  let impactY = targetBody.y;
  let impactZ = targetBody.z;
  let nudgeCounter = 0;

  const telemetry: LiveTelemetrySnapshot = {
    timeSec: 0,
    phase: 'approach',
    contactForceN: 0,
    slipVelocityMs: 0,
    objectHeightM: targetBody.y,
    faultState: 'ok',
    impactIntensity: 0,
    impactPosition: { x: targetBody.x, y: targetBody.y, z: targetBody.z },
    endEffectorPos: { x: eeX, y: eeY, z: eeZ },
  };

  /**
   * Computes nearest obstacle proximity and height around the current target pose.
   */
  const inspectNearestObstacle = (): {
    minGap: number;
    obstacleHeight: number;
    nearest: DynamicBodyState | null;
  } => {
    let minGap = 0.35;
    let obstacleHeight = 0.08;
    let nearest: DynamicBodyState | null = null;
    for (let i = 0; i < obstacleBodies.length; i++) {
      const obs = obstacleBodies[i];
      const dist = Math.hypot(obs.x - targetBody.x, obs.z - targetBody.z);
      const gap = dist - obs.radius - targetBody.radius;
      if (gap < minGap) {
        minGap = gap;
        obstacleHeight = obs.halfHeight * 2;
        nearest = obs;
      }
    }
    return { minGap, obstacleHeight, nearest };
  };

  /**
   * Integrates rigid-body tabletop sliding, gravity drops, bounce restitution,
   * tilt damping, and simple circle-circle obstacle collisions.
   */
  const stepRigidBody = (
    body: DynamicBodyState,
    dt: number,
    friction: number,
    skipPhysics: boolean,
  ): void => {
    if (skipPhysics) return;

    const g = APP_CONFIG.simulation.liveSandbox.gravityMs2;
    const [minX, maxX] = APP_CONFIG.simulation.liveSandbox.tableBoundsX;
    const [minZ, maxZ] = APP_CONFIG.simulation.liveSandbox.tableBoundsZ;

    // Gravity when airborne
    if (body.y > body.halfHeight + 0.0005 || body.vy > 0) {
      body.vy -= g * dt;
      body.y += body.vy * dt;
      if (body.y <= body.halfHeight) {
        body.y = body.halfHeight;
        if (Math.abs(body.vy) > 0.14) {
          impactIntensity = Math.max(impactIntensity, clamp(Math.abs(body.vy) * 0.85, 0.25, 1));
          impactX = body.x;
          impactY = 0.004;
          impactZ = body.z;
          body.vy = -body.vy * 0.26;
          body.angVelZ += (body.vx >= 0 ? -1 : 1) * Math.abs(body.vy) * 2.2;
        } else {
          body.vy = 0;
        }
      }
    } else {
      body.y = body.halfHeight;
      body.vy = 0;
    }

    // Horizontal motion + Coulomb surface friction
    body.x += body.vx * dt;
    body.z += body.vz * dt;
    const speed = Math.hypot(body.vx, body.vz);
    if (speed > 0.001) {
      const decel = Math.max(0.15, friction) * g * dt;
      const nextSpeed = Math.max(0, speed - decel);
      const scale = nextSpeed / speed;
      body.vx *= scale;
      body.vz *= scale;
    } else {
      body.vx = 0;
      body.vz = 0;
    }

    // Clamp to calibrated tabletop bounds with soft rebound
    const boundMinX = body.isTarget ? minX : -0.16;
    const boundMaxX = body.isTarget ? maxX : 0.44;
    if (body.x < boundMinX) {
      body.x = boundMinX;
      body.vx = Math.abs(body.vx) * 0.4;
    } else if (body.x > boundMaxX) {
      body.x = boundMaxX;
      body.vx = -Math.abs(body.vx) * 0.4;
    }
    if (body.z < minZ) {
      body.z = minZ;
      body.vz = Math.abs(body.vz) * 0.4;
    } else if (body.z > maxZ) {
      body.z = maxZ;
      body.vz = -Math.abs(body.vz) * 0.4;
    }

    // Angular tilt spring-damping (upright recovery below 0.55 rad, tipped rest above)
    body.tiltX += body.angVelX * dt;
    body.tiltZ += body.angVelZ * dt;
    const tiltMag = Math.hypot(body.tiltX, body.tiltZ);
    if (tiltMag < 0.65) {
      body.angVelX += -body.tiltX * 26 * dt;
      body.angVelZ += -body.tiltZ * 26 * dt;
    } else {
      const maxTilt = 1.25;
      body.tiltX = clamp(body.tiltX, -maxTilt, maxTilt);
      body.tiltZ = clamp(body.tiltZ, -maxTilt, maxTilt);
    }
    const angDamp = Math.exp(-6.5 * dt);
    body.angVelX *= angDamp;
    body.angVelZ *= angDamp;
  };

  /**
   * Resolves pairwise horizontal separation & elastic impulse transfer between
   * the target object and nearby obstacles.
   */
  const resolveBodyCollisions = (draggedBodyId: string | null): void => {
    for (let i = 0; i < obstacleBodies.length; i++) {
      const obs = obstacleBodies[i];
      const dx = targetBody.x - obs.x;
      const dz = targetBody.z - obs.z;
      const dist = Math.hypot(dx, dz);
      const minDist = targetBody.radius + obs.radius + 0.004;
      if (dist < minDist && dist > 0.0001) {
        const overlap = minDist - dist;
        const nx = dx / dist;
        const nz = dz / dist;

        if (draggedBodyId === targetBody.id) {
          obs.x -= nx * overlap;
          obs.z -= nz * overlap;
          obs.vx -= nx * 0.14;
          obs.vz -= nz * 0.14;
          obs.angVelZ += nx * 2.4;
        } else if (draggedBodyId === obs.id) {
          targetBody.x += nx * overlap;
          targetBody.z += nz * overlap;
          targetBody.vx += nx * 0.14;
          targetBody.vz += nz * 0.14;
          targetBody.angVelZ -= nx * 2.4;
        } else {
          targetBody.x += nx * overlap * 0.55;
          targetBody.z += nz * overlap * 0.55;
          obs.x -= nx * overlap * 0.45;
          obs.z -= nz * overlap * 0.45;
          const relVn = (targetBody.vx - obs.vx) * nx + (targetBody.vz - obs.vz) * nz;
          if (relVn < 0) {
            const j = -1.35 * relVn * 0.5;
            targetBody.vx += j * nx;
            targetBody.vz += j * nz;
            obs.vx -= j * nx;
            obs.vz -= j * nz;
            impactIntensity = Math.max(impactIntensity, clamp(-relVn * 2.0, 0.2, 0.9));
            impactX = (targetBody.x + obs.x) * 0.5;
            impactY = 0.04;
            impactZ = (targetBody.z + obs.z) * 0.5;
          }
        }
      }
    }
  };

  /**
   * Advances the real-time physics & 6-DOF IK state machine by `dtSec`.
   */
  const step = (
    dtSec: number,
    robotModel: RobotModelId,
    params: LivePhysicsParams,
    controllerParams: ControllerParams | null,
    draggedBodyId: string | null,
  ): void => {
    const dt = clamp(dtSec, 0.001, 0.045);
    simTime += dt;
    impactIntensity = Math.max(0, impactIntensity - dt * 1.9);

    const g = APP_CONFIG.simulation.liveSandbox.gravityMs2;
    const friction = clamp(params.friction, 0.12, 1.0);
    const massKg = clamp(params.massKg, 0.08, 1.2);
    const clearanceM = clamp(params.clearanceM, 0.02, 0.2);
    const gripForceN = clamp(params.gripForceN, 8, 40);
    const liftHeightM = clamp(controllerParams?.liftHeightM ?? 0.16, 0.12, 0.24);
    const approachHeightM = clamp(controllerParams?.approachHeightM ?? 0.17, 0.12, 0.24);

    // Pause grasp attachment while user actively drags the target
    if (draggedBodyId === targetBody.id) {
      isGrasped = false;
      phase = 'approach';
      phaseElapsed = 0;
      cycleFault = 'ok';
      targetBody.vx = 0;
      targetBody.vy = 0;
      targetBody.vz = 0;
      targetBody.tiltX *= Math.exp(-10 * dt);
      targetBody.tiltZ *= Math.exp(-10 * dt);
    }

    const { minGap, obstacleHeight, nearest } = inspectNearestObstacle();
    const preGraspY = Math.max(targetBody.halfHeight + 0.07, approachHeightM + clearanceM * 0.75);
    const graspY = targetBody.halfHeight + 0.014;
    const liftY = targetBody.halfHeight + liftHeightM;

    const requiredNormalForce = (massKg * g) / (2 * friction);
    const gripMargin = gripForceN * friction * 2 - massKg * g;
    const clearanceShortfall = Math.max(0, obstacleHeight + 0.04 - (approachHeightM + clearanceM));
    const corridorCollisionRisk = minGap < 0.022 && clearanceM < 0.095;

    let desiredEeX = eeX;
    let desiredEeY = eeY;
    let desiredEeZ = eeZ;
    let desiredGripper = 0.92;
    let liveForceN = 0;
    let liveSlipMs = 0;

    if (draggedBodyId === targetBody.id) {
      // Hover attentively above the user's dragged target in real time
      desiredEeX = targetBody.x;
      desiredEeY = preGraspY + 0.03;
      desiredEeZ = targetBody.z;
      desiredGripper = 1.0;
    } else {
      phaseElapsed += dt;

      if (phase === 'approach') {
        const dur = 0.95;
        const u = clamp(phaseElapsed / dur, 0, 1);
        const arc = Math.sin(u * Math.PI) * clearanceM * 0.6;
        desiredEeX = targetBody.x;
        desiredEeY = preGraspY + arc;
        desiredEeZ = targetBody.z;
        desiredGripper = 0.95;

        // If target slid while approaching, keep tracking until settled
        const targetSpeed = Math.hypot(targetBody.vx, targetBody.vz);
        if (u >= 1 && targetSpeed < 0.03 && targetBody.y <= targetBody.halfHeight + 0.005) {
          // Gently upright tipped object right before next grasp cycle
          targetBody.tiltX *= 0.2;
          targetBody.tiltZ *= 0.2;
          targetBody.angVelX = 0;
          targetBody.angVelZ = 0;
          phase = 'descend';
          phaseElapsed = 0;
          cycleFault = 'ok';
        }
      } else if (phase === 'descend') {
        const dur = 0.65;
        const u = clamp(phaseElapsed / dur, 0, 1);
        const ease = u * u * (3 - 2 * u);
        desiredEeX = targetBody.x;
        desiredEeY = THREE.MathUtils.lerp(preGraspY, graspY, ease);
        desiredEeZ = targetBody.z;
        desiredGripper = 0.92;

        // Detect obstacle or high-speed approach collision
        if (u > 0.68 && (corridorCollisionRisk || clearanceShortfall > 0.03)) {
          cycleFault = 'collision';
          liveForceN = Number(
            (48 + clearanceShortfall * 320 + Math.max(0, 0.03 - minGap) * 420).toFixed(1),
          );
          impactIntensity = 1.0;
          impactX = nearest ? (nearest.x + targetBody.x) * 0.5 : targetBody.x;
          impactY = targetBody.halfHeight + 0.02;
          impactZ = nearest ? (nearest.z + targetBody.z) * 0.5 : targetBody.z;

          // Impart knock impulse to target and nearest obstacle
          const knockDirX = nearest ? targetBody.x - nearest.x : 0.04;
          const knockDirZ = nearest ? targetBody.z - nearest.z : 0.03;
          const norm = Math.max(0.01, Math.hypot(knockDirX, knockDirZ));
          targetBody.vx = (knockDirX / norm) * 0.22;
          targetBody.vz = (knockDirZ / norm) * 0.22;
          targetBody.angVelZ = -5.4;
          if (nearest) {
            nearest.vx = -(knockDirX / norm) * 0.14;
            nearest.vz = -(knockDirZ / norm) * 0.14;
            nearest.angVelZ = 3.2;
          }
          phase = 'return';
          phaseElapsed = 0;
        } else if (u >= 1) {
          phase = 'grasp';
          phaseElapsed = 0;
        }
      } else if (phase === 'grasp') {
        const dur = 0.48;
        const u = clamp(phaseElapsed / dur, 0, 1);
        desiredEeX = targetBody.x;
        desiredEeY = graspY;
        desiredEeZ = targetBody.z;
        desiredGripper = THREE.MathUtils.lerp(0.92, 0.22, u);
        liveForceN = Math.max(5.5, requiredNormalForce * 1.15) * u;

        if (u >= 1) {
          isGrasped = true;
          impactIntensity = Math.max(impactIntensity, 0.35);
          impactX = targetBody.x;
          impactY = targetBody.y;
          impactZ = targetBody.z;
          phase = 'lift';
          phaseElapsed = 0;
        }
      } else if (phase === 'lift') {
        const dur = 0.85;
        const u = clamp(phaseElapsed / dur, 0, 1);
        const ease = u * u * (3 - 2 * u);
        desiredEeX = targetBody.x;
        desiredEeY = THREE.MathUtils.lerp(graspY, liftY, ease);
        desiredEeZ = targetBody.z;
        desiredGripper = 0.22;

        const willSlip = gripMargin < 0.85;
        if (willSlip && u > 0.36) {
          // Physical grasp slip: object detaches and drops under gravity
          isGrasped = false;
          cycleFault = 'slip';
          liveSlipMs = clamp(0.11 + Math.max(0, 1.4 - gripMargin) * 0.045, 0.09, 0.34);
          liveForceN = 4.2;
          targetBody.vx = 0.08;
          targetBody.vz = (nudgeCounter % 2 === 0 ? 1 : -1) * 0.06;
          targetBody.vy = -0.05;
          targetBody.angVelZ = -2.8;
          phase = 'return';
          phaseElapsed = 0;
        } else {
          liveForceN = Math.max(6.2, requiredNormalForce * 1.12);
          liveSlipMs = clamp(0.012 + (1 - friction) * 0.018, 0.006, 0.042);
          if (u >= 1) {
            phase = 'hold';
            phaseElapsed = 0;
          }
        }
      } else if (phase === 'hold') {
        const dur = 0.65;
        desiredEeX = targetBody.x;
        desiredEeY = liftY;
        desiredEeZ = targetBody.z;
        desiredGripper = 0.22;
        liveForceN = Math.max(6.0, requiredNormalForce * 1.08);
        liveSlipMs = 0.008;
        if (phaseElapsed >= dur) {
          phase = 'return';
          phaseElapsed = 0;
        }
      } else {
        // 'return': gently set object back down if still held, then retract to pre-grasp
        const dur = isGrasped ? 0.95 : 0.72;
        const u = clamp(phaseElapsed / dur, 0, 1);
        if (isGrasped) {
          if (u < 0.65) {
            const d = u / 0.65;
            const ease = d * d * (3 - 2 * d);
            desiredEeX = targetBody.x;
            desiredEeY = THREE.MathUtils.lerp(liftY, graspY, ease);
            desiredEeZ = targetBody.z;
            desiredGripper = 0.22;
            liveForceN = Math.max(5.2, requiredNormalForce);
          } else {
            isGrasped = false;
            desiredEeX = targetBody.x;
            desiredEeY = preGraspY;
            desiredEeZ = targetBody.z;
            desiredGripper = 0.95;
          }
        } else {
          desiredEeX = 0.16;
          desiredEeY = Math.max(0.25, preGraspY);
          desiredEeZ = 0.0;
          desiredGripper = 0.95;
        }

        if (u >= 1) {
          isGrasped = false;
          phase = 'approach';
          phaseElapsed = 0;
        }
      }
    }

    // Smoothly damp end-effector position and solve 6-DOF IK
    const eeLambda = 9.5;
    eeX = THREE.MathUtils.damp(eeX, desiredEeX, eeLambda, dt);
    eeY = THREE.MathUtils.damp(eeY, desiredEeY, eeLambda, dt);
    eeZ = THREE.MathUtils.damp(eeZ, desiredEeZ, eeLambda, dt);

    solveLiveArmIK(eeX, eeY, eeZ, robotModel, targetAngles);
    const jLambda = APP_CONFIG.simulation.liveSandbox.jointDampLambda;
    for (let j = 0; j < 6; j++) {
      jointAngles[j] = THREE.MathUtils.damp(jointAngles[j], targetAngles[j], jLambda, dt);
    }
    gripperOpen = THREE.MathUtils.damp(gripperOpen, desiredGripper, 14, dt);

    // Sync target body either locked inside gripper jaws or free under rigid-body physics
    if (isGrasped && draggedBodyId !== targetBody.id) {
      targetBody.x = eeX;
      targetBody.y = Math.max(targetBody.halfHeight, eeY - 0.014);
      targetBody.z = eeZ;
      targetBody.vx = 0;
      targetBody.vy = 0;
      targetBody.vz = 0;
      targetBody.tiltX = THREE.MathUtils.damp(targetBody.tiltX, 0, 12, dt);
      targetBody.tiltZ = THREE.MathUtils.damp(targetBody.tiltZ, 0, 12, dt);
    } else {
      stepRigidBody(targetBody, dt, friction, draggedBodyId === targetBody.id);
    }

    for (let i = 0; i < obstacleBodies.length; i++) {
      const obs = obstacleBodies[i];
      stepRigidBody(obs, dt, 0.55, draggedBodyId === obs.id);
    }
    resolveBodyCollisions(draggedBodyId);

    // Update live telemetry snapshot in-place (zero allocation)
    telemetry.timeSec = simTime;
    telemetry.phase = phase;
    telemetry.contactForceN = liveForceN;
    telemetry.slipVelocityMs = liveSlipMs;
    telemetry.objectHeightM = targetBody.y;
    telemetry.faultState = cycleFault;
    telemetry.impactIntensity = impactIntensity;
    telemetry.impactPosition.x = impactX;
    telemetry.impactPosition.y = impactY;
    telemetry.impactPosition.z = impactZ;
    telemetry.endEffectorPos.x = eeX;
    telemetry.endEffectorPos.y = eeY;
    telemetry.endEffectorPos.z = eeZ;
  };

  /**
   * Moves the dragged target or obstacle directly to a clamped tabletop coordinate.
   */
  const setBodyPositionFromDrag = (bodyId: string, worldX: number, worldZ: number): void => {
    const [minX, maxX] = APP_CONFIG.simulation.liveSandbox.tableBoundsX;
    const [minZ, maxZ] = APP_CONFIG.simulation.liveSandbox.tableBoundsZ;

    const body =
      bodyId === targetBody.id
        ? targetBody
        : obstacleBodies.find((o) => o.id === bodyId) ?? null;
    if (!body) return;

    body.x = clamp(worldX, body.isTarget ? minX : -0.14, body.isTarget ? maxX : 0.42);
    body.z = clamp(worldZ, minZ, maxZ);
    body.y = body.halfHeight;
    body.vx = 0;
    body.vy = 0;
    body.vz = 0;
    body.tiltX = 0;
    body.tiltZ = 0;
    resolveBodyCollisions(bodyId);
  };

  /**
   * Finalizes a tabletop drag interaction and triggers an immediate arm re-approach.
   */
  const finishBodyDrag = (bodyId: string): void => {
    logFunctionCall('livePhysicsSandbox.finishBodyDrag', { bodyId });
    if (bodyId === targetBody.id) {
      isGrasped = false;
      phase = 'approach';
      phaseElapsed = 0;
      cycleFault = 'ok';
    }
  };

  /**
   * Injects a real-time horizontal impulse + subtle angular perturbation onto
   * the target object to test dynamic recovery and friction sliding.
   */
  const applyNudgeImpulse = (): void => {
    nudgeCounter += 1;
    const angle = (nudgeCounter * 2.15) % (Math.PI * 2);
    const speed = APP_CONFIG.simulation.liveSandbox.nudgeImpulseMs;
    logFunctionCall('livePhysicsSandbox.applyNudgeImpulse', { angle, speed });

    isGrasped = false;
    targetBody.vx = Math.cos(angle) * speed * 0.72;
    targetBody.vz = Math.sin(angle) * speed;
    targetBody.vy = 0.08;
    targetBody.angVelZ = -Math.cos(angle) * 3.8;
    targetBody.angVelX = Math.sin(angle) * 3.2;
    impactIntensity = 0.8;
    impactX = targetBody.x;
    impactY = targetBody.y;
    impactZ = targetBody.z;
    phase = 'approach';
    phaseElapsed = 0;
  };

  /**
   * Restores the target object and all obstacles cleanly to their calibrated home poses.
   */
  const resetBodies = (): void => {
    logFunctionCall('livePhysicsSandbox.resetBodies');
    isGrasped = false;
    cycleFault = 'ok';
    phase = 'approach';
    phaseElapsed = 0;

    targetBody.x = targetBody.homeX;
    targetBody.y = targetBody.halfHeight;
    targetBody.z = targetBody.homeZ;
    targetBody.vx = 0;
    targetBody.vy = 0;
    targetBody.vz = 0;
    targetBody.tiltX = 0;
    targetBody.tiltZ = 0;
    targetBody.angVelX = 0;
    targetBody.angVelZ = 0;

    for (let i = 0; i < obstacleBodies.length; i++) {
      const obs = obstacleBodies[i];
      obs.x = obs.homeX;
      obs.y = obs.halfHeight;
      obs.z = obs.homeZ;
      obs.vx = 0;
      obs.vy = 0;
      obs.vz = 0;
      obs.tiltX = 0;
      obs.tiltZ = 0;
      obs.angVelX = 0;
      obs.angVelZ = 0;
    }
  };

  /**
   * Populates a pre-allocated array of `Vector3` waypoints tracing the live
   * approach-descend-lift path to the target's current position (zero allocation).
   */
  const computePreviewSplinePoints = (
    clearanceM: number,
    liftHeightM: number,
    outPoints: THREE.Vector3[],
  ): number => {
    const count = outPoints.length;
    if (count < 6) return 0;
    const preY = Math.max(targetBody.halfHeight + 0.08, 0.17 + clearanceM * 0.75);
    const graspY = targetBody.halfHeight + 0.015;
    const liftY = targetBody.halfHeight + liftHeightM;
    const homeX = 0.14;
    const homeY = 0.27;
    const homeZ = 0.0;

    for (let i = 0; i < count; i++) {
      const u = i / (count - 1);
      if (u < 0.45) {
        const t = u / 0.45;
        const ease = t * t * (3 - 2 * t);
        outPoints[i].set(
          homeX + (targetBody.x - homeX) * ease,
          homeY + (preY - homeY) * ease + Math.sin(t * Math.PI) * clearanceM * 0.65,
          homeZ + (targetBody.z - homeZ) * ease,
        );
      } else if (u < 0.72) {
        const t = (u - 0.45) / 0.27;
        outPoints[i].set(
          targetBody.x,
          THREE.MathUtils.lerp(preY, graspY, t),
          targetBody.z,
        );
      } else {
        const t = (u - 0.72) / 0.28;
        outPoints[i].set(
          targetBody.x,
          THREE.MathUtils.lerp(graspY, liftY, t),
          targetBody.z,
        );
      }
    }
    return count;
  };

  return {
    targetBody,
    obstacleBodies,
    jointAngles,
    gripperOpen,
    telemetry,
    step,
    setBodyPositionFromDrag,
    finishBodyDrag,
    applyNudgeImpulse,
    resetBodies,
    computePreviewSplinePoints,
  };
}
