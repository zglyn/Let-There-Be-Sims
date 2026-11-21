/**
 * Procedural 3D Articulated Robot Rig Builder (SO-ARM101 & Franka Panda)
 *
 * Constructs a multi-part composite PBR robot arm hierarchy (`THREE.Group` with
 * 12+ distinct mechanical sub-assemblies) grounded cleanly at bottom-center on
 * the tabletop origin `(0, 0, 0)`. Varied material roughness and metalness
 * separate machined aluminum housings, matte carbon-polymer link shells, dark
 * elastomer gripper pads, and amber safety calibration rings.
 *
 * Use cases:
 * - Real-time kinematic replay of 60-frame trajectories emitted by the physics
 *   evaluator or Cloud MuJoCo swarm inside `RobotStage3D`.
 * - Instant switching between the open-source SO-ARM101 (LeRobot) and Franka
 *   Emika Panda visual profiles.
 */

import * as THREE from 'three';
import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import { logFunctionCall } from '../utils/logger';

export interface RobotRigHandles {
  root: THREE.Group;
  baseYawPivot: THREE.Group;
  shoulderPivot: THREE.Group;
  elbowPivot: THREE.Group;
  wristPitchPivot: THREE.Group;
  wristRollPivot: THREE.Group;
  leftFinger: THREE.Group;
  rightFinger: THREE.Group;
  toolTipMarker: THREE.Object3D;
  wristCameraAnchor: THREE.Object3D;
  wristLookTarget: THREE.Object3D;
  applyJointState: (angles: [number, number, number, number, number, number], gripperOpen: number) => void;
  dispose: () => void;
}

/**
 * Builds the composite articulated 6-DOF robot arm hierarchy for the requested
 * robot model and returns kinematic joint handles plus a full GPU disposer.
 */
export function buildRobotArmRig(robotModel: RobotModelId): RobotRigHandles {
  logFunctionCall('buildRobotArmRig', { robotModel });
  const spec = APP_CONFIG.robots[robotModel];
  const isPanda = robotModel === 'franka_panda';

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];

  const trackGeo = <T extends THREE.BufferGeometry>(g: T): T => {
    geometries.push(g);
    return g;
  };
  const trackMat = <T extends THREE.Material>(m: T): T => {
    materials.push(m);
    return m;
  };

  // Distinct PBR material palette (varying roughness >= 0.25 & metalness across adjacent parts)
  const matteHousingMat = trackMat(
    new THREE.MeshStandardMaterial({
      color: isPanda ? 0xe5e9f0 : 0x222938,
      roughness: 0.62,
      metalness: 0.12,
    }),
  );
  const brushedAluminumMat = trackMat(
    new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      roughness: 0.24,
      metalness: 0.88,
    }),
  );
  const darkServoMat = trackMat(
    new THREE.MeshStandardMaterial({
      color: 0x11151f,
      roughness: 0.48,
      metalness: 0.35,
    }),
  );
  const accentRingMat = trackMat(
    new THREE.MeshStandardMaterial({
      color: isPanda ? 0x06b6d4 : 0xf59e0b,
      roughness: 0.3,
      metalness: 0.6,
    }),
  );
  const elastomerPadMat = trackMat(
    new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.92,
      metalness: 0.02,
    }),
  );

  const root = new THREE.Group();
  root.name = `robot_rig_${robotModel}`;

  // 1. Anchored Base Mounting Plate (bottom flush at y = 0)
  const flangeH = 0.016;
  const flangeMesh = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.068, 0.074, flangeH, 28)),
    brushedAluminumMat,
  );
  flangeMesh.position.y = flangeH / 2;
  flangeMesh.castShadow = true;
  flangeMesh.receiveShadow = true;
  root.add(flangeMesh);

  // Mounting bolts (4 instanced-style socket cap screws)
  const boltGeo = trackGeo(new THREE.CylinderGeometry(0.005, 0.005, 0.008, 10));
  for (let b = 0; b < 4; b++) {
    const theta = (b * Math.PI) / 2 + Math.PI / 4;
    const bolt = new THREE.Mesh(boltGeo, darkServoMat);
    bolt.position.set(Math.cos(theta) * 0.056, flangeH + 0.003, Math.sin(theta) * 0.056);
    root.add(bolt);
  }

  // Pedestal column
  const pedestalH = spec.baseHeightM - flangeH;
  const pedestalMesh = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.044, 0.052, pedestalH, 24)),
    matteHousingMat,
  );
  pedestalMesh.position.y = flangeH + pedestalH / 2;
  pedestalMesh.castShadow = true;
  pedestalMesh.receiveShadow = true;
  root.add(pedestalMesh);

  // Calibration collar ring
  const collarRing = new THREE.Mesh(
    trackGeo(new THREE.TorusGeometry(0.045, 0.004, 12, 32)),
    accentRingMat,
  );
  collarRing.rotation.x = Math.PI / 2;
  collarRing.position.y = spec.baseHeightM - 0.004;
  root.add(collarRing);

  // 2. Joint 1 — Base Yaw Turret
  const baseYawPivot = new THREE.Group();
  baseYawPivot.position.y = spec.baseHeightM;
  root.add(baseYawPivot);

  const turretBlock = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(0.066, 0.046, 0.058)),
    darkServoMat,
  );
  turretBlock.position.y = 0.023;
  turretBlock.castShadow = true;
  baseYawPivot.add(turretBlock);

  const shoulderDrum = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.026, 0.026, 0.072, 20)),
    brushedAluminumMat,
  );
  shoulderDrum.rotation.x = Math.PI / 2;
  shoulderDrum.position.y = 0.032;
  shoulderDrum.castShadow = true;
  baseYawPivot.add(shoulderDrum);

  // 3. Joint 2 — Shoulder Pitch & Upper Arm Assembly
  const shoulderPivot = new THREE.Group();
  shoulderPivot.position.y = 0.032;
  baseYawPivot.add(shoulderPivot);

  const l1 = spec.upperArmM;
  // Twin structural aluminum struts + central polymer spine
  const strutGeo = trackGeo(new THREE.BoxGeometry(l1, 0.016, 0.008));
  const leftStrut = new THREE.Mesh(strutGeo, brushedAluminumMat);
  leftStrut.position.set(l1 / 2, 0, 0.022);
  leftStrut.castShadow = true;
  shoulderPivot.add(leftStrut);

  const rightStrut = new THREE.Mesh(strutGeo, brushedAluminumMat);
  rightStrut.position.set(l1 / 2, 0, -0.022);
  rightStrut.castShadow = true;
  shoulderPivot.add(rightStrut);

  const upperCore = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(l1 * 0.84, 0.028, 0.032)),
    matteHousingMat,
  );
  upperCore.position.set(l1 / 2, 0, 0);
  upperCore.castShadow = true;
  shoulderPivot.add(upperCore);

  // Elbow servo drum
  const elbowDrum = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.022, 0.022, 0.06, 20)),
    darkServoMat,
  );
  elbowDrum.rotation.x = Math.PI / 2;
  elbowDrum.position.set(l1, 0, 0);
  elbowDrum.castShadow = true;
  shoulderPivot.add(elbowDrum);

  const elbowTrim = new THREE.Mesh(
    trackGeo(new THREE.TorusGeometry(0.022, 0.003, 10, 24)),
    accentRingMat,
  );
  elbowTrim.position.set(l1, 0, 0.031);
  shoulderPivot.add(elbowTrim);

  // 4. Joint 3 — Elbow Pitch & Forearm Assembly
  const elbowPivot = new THREE.Group();
  elbowPivot.position.set(l1, 0, 0);
  shoulderPivot.add(elbowPivot);

  const l2 = spec.forearmM;
  const forearmTube = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.016, 0.019, l2, 18)),
    matteHousingMat,
  );
  forearmTube.rotation.z = -Math.PI / 2;
  forearmTube.position.set(l2 / 2, 0, 0);
  forearmTube.castShadow = true;
  elbowPivot.add(forearmTube);

  // Cable guide rib along forearm
  const cableRib = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(l2 * 0.72, 0.009, 0.01)),
    darkServoMat,
  );
  cableRib.position.set(l2 * 0.5, 0.018, 0);
  elbowPivot.add(cableRib);

  // 5. Joint 4 — Wrist Pitch Gimbal
  const wristPitchPivot = new THREE.Group();
  wristPitchPivot.position.set(l2, 0, 0);
  elbowPivot.add(wristPitchPivot);

  const wristJointDrum = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.016, 0.016, 0.044, 16)),
    brushedAluminumMat,
  );
  wristJointDrum.rotation.x = Math.PI / 2;
  wristJointDrum.castShadow = true;
  wristPitchPivot.add(wristJointDrum);

  const wristStem = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.013, 0.014, 0.036, 16)),
    darkServoMat,
  );
  wristStem.rotation.z = -Math.PI / 2;
  wristStem.position.set(0.02, 0, 0);
  wristPitchPivot.add(wristStem);

  // 6. Joint 5 — Wrist Roll & Parallel Gripper Carriage
  const wristRollPivot = new THREE.Group();
  wristRollPivot.position.set(0.042, 0, 0);
  wristPitchPivot.add(wristRollPivot);

  const gripperPalm = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(0.02, 0.024, 0.068)),
    brushedAluminumMat,
  );
  gripperPalm.castShadow = true;
  wristRollPivot.add(gripperPalm);

  const palmAccent = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(0.022, 0.005, 0.046)),
    accentRingMat,
  );
  palmAccent.position.y = 0.011;
  wristRollPivot.add(palmAccent);

  // Dual Parallel Gripper Fingers
  const leftFinger = new THREE.Group();
  const rightFinger = new THREE.Group();

  const buildFinger = (side: 1 | -1): THREE.Group => {
    const fg = new THREE.Group();
    const jaw = new THREE.Mesh(
      trackGeo(new THREE.BoxGeometry(0.046, 0.014, 0.008)),
      matteHousingMat,
    );
    jaw.position.set(0.028, 0, 0);
    jaw.castShadow = true;
    fg.add(jaw);

    const pad = new THREE.Mesh(
      trackGeo(new THREE.BoxGeometry(0.028, 0.016, 0.005)),
      elastomerPadMat,
    );
    pad.position.set(0.036, 0, -side * 0.005);
    fg.add(pad);
    return fg;
  };

  leftFinger.add(buildFinger(1));
  rightFinger.add(buildFinger(-1));
  leftFinger.position.set(0, 0, 0.024);
  rightFinger.position.set(0, 0, -0.024);
  wristRollPivot.add(leftFinger);
  wristRollPivot.add(rightFinger);

  const toolTipMarker = new THREE.Object3D();
  toolTipMarker.position.set(0.046, 0, 0);
  wristRollPivot.add(toolTipMarker);

  // Wrist-mounted Eye-in-Hand Optical Camera Module (standard on SO-ARM101 & Panda setups)
  const wristCamHousing = new THREE.Mesh(
    trackGeo(new THREE.BoxGeometry(0.022, 0.014, 0.026)),
    darkServoMat,
  );
  wristCamHousing.position.set(0.008, 0.022, 0);
  wristCamHousing.castShadow = true;
  wristRollPivot.add(wristCamHousing);

  const wristCamLens = new THREE.Mesh(
    trackGeo(new THREE.CylinderGeometry(0.0045, 0.005, 0.006, 16)),
    trackMat(
      new THREE.MeshStandardMaterial({
        color: 0x06b6d4,
        roughness: 0.14,
        metalness: 0.85,
      }),
    ),
  );
  wristCamLens.rotation.z = Math.PI / 2;
  wristCamLens.position.set(0.02, 0.022, 0);
  wristRollPivot.add(wristCamLens);

  // Eye-in-Hand Camera Anchor positioned just ahead of the wrist optical module
  const wristCameraAnchor = new THREE.Object3D();
  wristCameraAnchor.position.set(-0.025, 0.055, 0);
  wristRollPivot.add(wristCameraAnchor);

  const wristLookTarget = new THREE.Object3D();
  wristLookTarget.position.set(0.16, -0.015, 0);
  wristRollPivot.add(wristLookTarget);

  /**
   * Applies a 6-DOF joint angle vector (radians) and normalized gripper
   * aperture [0..1] directly to the Three.js scene graph pivots.
   */
  const applyJointState = (
    angles: [number, number, number, number, number, number],
    gripperOpen: number,
  ): void => {
    const [baseYaw, shoulderPitch, elbowPitch, wristPitch, wristRoll] = angles;
    baseYawPivot.rotation.y = -baseYaw;
    shoulderPivot.rotation.z = shoulderPitch;
    elbowPivot.rotation.z = elbowPitch;
    wristPitchPivot.rotation.z = wristPitch;
    wristRollPivot.rotation.x = wristRoll;

    const halfAperture = 0.012 + Math.max(0, Math.min(1, gripperOpen)) * 0.022;
    leftFinger.position.z = halfAperture;
    rightFinger.position.z = -halfAperture;
  };

  // Default calibrated standby pose
  applyJointState([0.12, 1.05, -1.45, -1.15, 0, 0], 0.85);

  /**
   * Disposes all allocated robot geometries and PBR materials on unmount.
   */
  const dispose = (): void => {
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
  };

  return {
    root,
    baseYawPivot,
    shoulderPivot,
    elbowPivot,
    wristPitchPivot,
    wristRollPivot,
    leftFinger,
    rightFinger,
    toolTipMarker,
    wristCameraAnchor,
    wristLookTarget,
    applyJointState,
    dispose,
  };
}
