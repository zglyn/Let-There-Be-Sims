/**
 * Multi-Viewport Three.js Robotic Arm Stage Renderer (`stageViewportRenderer.ts`)
 *
 * Implements single-context split-screen WebGL rendering using `setScissorTest`
 * and `setViewport` so the Sim2Real Compiler can render two live synchronized
 * robotic arm views simultaneously without allocating a second WebGL context:
 *
 * 1. `dual_cam` (Default):
 *    - Left viewport: Interactive orbitable Workcell 3D camera.
 *    - Right viewport: Secondary inspection camera — either a real-time
 *      **Wrist Eye-in-Hand Camera** locked to the robot's `wristCameraAnchor`
 *      (looking straight down the gripper jaws toward the target object), or
 *      a calibrated `top` / `side` / `gripper` camera. When `hideRobotArm` is
 *      active (Scene Map Mode), automatically displays the Top-Down Calibration
 *      camera in the right viewport.
 * 2. `dual_arms`:
 *    - Left viewport: **SO-ARM101 (6-DOF)** executing the current rollout or
 *      standby kinematics.
 *    - Right viewport: **Franka Emika Panda (7-DOF)** executing the exact same
 *      synchronized trajectory on the identical calibrated workcell from the
 *      same live orbit camera angle for direct kinematic comparison.
 * 3. `single`:
 *    - Full-bleed single 3D viewport.
 *
 * Use cases:
 * - Inspecting both third-person workcell clearance and first-person wrist
 *   gripper alignment simultaneously (mirroring LeRobot / BotFails dual cams).
 * - Comparing SO-ARM101 and Franka Emika Panda reach & clearance side-by-side
 *   on the same reconstructed scene without switching tabs.
 */

import * as THREE from 'three';
import {
  APP_CONFIG,
  type SecondaryCameraModeId,
  type StageViewModeId,
} from '../config/appConfig';
import type { RobotRigHandles } from './robotRigBuilder';

export interface DualViewportRenderParams {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  primaryCamera: THREE.PerspectiveCamera;
  secondaryCamera: THREE.PerspectiveCamera;
  primaryRig: RobotRigHandles | null;
  secondaryRig: RobotRigHandles | null;
  layoutMode: StageViewModeId;
  secondaryCamMode: SecondaryCameraModeId;
  hideRobotArm: boolean;
  viewportWidth: number;
  viewportHeight: number;
}

// Pre-allocated vectors to guarantee zero garbage collection inside the 60 FPS animation loop
const _wristPos = new THREE.Vector3();
const _wristLook = new THREE.Vector3();
const _wristUp = new THREE.Vector3(0, 1, 0);
const _renderSize = new THREE.Vector2();

/**
 * Updates the secondary Three.js camera transform to match either the active
 * robot arm's live Wrist Eye-in-Hand mount or a calibrated preset angle.
 */
export function updateSecondaryCameraTransform(
  secondaryCamera: THREE.PerspectiveCamera,
  secondaryCamMode: SecondaryCameraModeId,
  primaryRig: RobotRigHandles | null,
  hideRobotArm: boolean,
): void {
  const effectiveMode: SecondaryCameraModeId =
    hideRobotArm && secondaryCamMode === 'wrist' ? 'top' : secondaryCamMode;

  if (effectiveMode === 'wrist' && primaryRig) {
    primaryRig.root.updateMatrixWorld(true);
    primaryRig.wristCameraAnchor.getWorldPosition(_wristPos);
    primaryRig.wristLookTarget.getWorldPosition(_wristLook);

    const targetFov = APP_CONFIG.secondaryCameraModes.wrist.fov;
    if (secondaryCamera.fov !== targetFov) {
      secondaryCamera.fov = targetFov;
    }
    secondaryCamera.position.copy(_wristPos);
    secondaryCamera.up.copy(_wristUp);
    secondaryCamera.lookAt(_wristLook);
    return;
  }

  const presetKey =
    effectiveMode === 'top'
      ? 'top'
      : effectiveMode === 'side'
        ? 'side'
        : 'gripper';
  const preset = APP_CONFIG.cameraPresets[presetKey];
  const targetFov = APP_CONFIG.secondaryCameraModes[effectiveMode].fov;
  if (secondaryCamera.fov !== targetFov) {
    secondaryCamera.fov = targetFov;
  }
  secondaryCamera.position.set(...preset.position);
  secondaryCamera.up.set(0, 1, 0);
  secondaryCamera.lookAt(...preset.target);
}

/**
 * Renders one frame of the Three.js robotic stage in either `single`,
 * `dual_cam`, or `dual_arms` layout using a single WebGLRenderer.
 */
export function renderRobotStageViewports(params: DualViewportRenderParams): void {
  const {
    renderer,
    scene,
    primaryCamera,
    secondaryCamera,
    primaryRig,
    secondaryRig,
    layoutMode,
    secondaryCamMode,
    hideRobotArm,
  } = params;

  renderer.getSize(_renderSize);
  const fullW = Math.max(2, Math.floor(_renderSize.x || params.viewportWidth));
  const fullH = Math.max(2, Math.floor(_renderSize.y || params.viewportHeight));

  if (layoutMode === 'single') {
    if (primaryRig) primaryRig.root.visible = !hideRobotArm;
    if (secondaryRig) secondaryRig.root.visible = false;

    const aspect = fullW / fullH;
    if (Math.abs(primaryCamera.aspect - aspect) > 0.001) {
      primaryCamera.aspect = aspect;
      primaryCamera.updateProjectionMatrix();
    }

    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, fullW, fullH);
    renderer.render(scene, primaryCamera);
    return;
  }

  const leftW = Math.floor(fullW / 2);
  const rightW = fullW - leftW;
  const leftAspect = leftW / fullH;
  const rightAspect = rightW / fullH;

  renderer.setScissorTest(true);

  // PASS 1 — Left Viewport (Workcell Perspective or SO-ARM101)
  if (primaryRig) primaryRig.root.visible = !hideRobotArm;
  if (secondaryRig) secondaryRig.root.visible = false;

  if (Math.abs(primaryCamera.aspect - leftAspect) > 0.001) {
    primaryCamera.aspect = leftAspect;
    primaryCamera.updateProjectionMatrix();
  }
  renderer.setViewport(0, 0, leftW, fullH);
  renderer.setScissor(0, 0, leftW, fullH);
  renderer.render(scene, primaryCamera);

  // PASS 2 — Right Viewport (Wrist Eye-in-Hand / Ortho Cam OR Second Robotic Arm)
  if (layoutMode === 'dual_arms' && secondaryRig && !hideRobotArm) {
    if (primaryRig) primaryRig.root.visible = false;
    secondaryRig.root.visible = true;

    if (Math.abs(primaryCamera.aspect - rightAspect) > 0.001) {
      primaryCamera.aspect = rightAspect;
      primaryCamera.updateProjectionMatrix();
    }
    renderer.setViewport(leftW, 0, rightW, fullH);
    renderer.setScissor(leftW, 0, rightW, fullH);
    renderer.render(scene, primaryCamera);

    // Restore visibility for external callers
    if (primaryRig) primaryRig.root.visible = !hideRobotArm;
    secondaryRig.root.visible = false;
  } else {
    if (primaryRig) primaryRig.root.visible = !hideRobotArm;
    if (secondaryRig) secondaryRig.root.visible = false;

    updateSecondaryCameraTransform(
      secondaryCamera,
      secondaryCamMode,
      primaryRig,
      hideRobotArm,
    );
    secondaryCamera.aspect = rightAspect;
    secondaryCamera.updateProjectionMatrix();

    renderer.setViewport(leftW, 0, rightW, fullH);
    renderer.setScissor(leftW, 0, rightW, fullH);
    renderer.render(scene, secondaryCamera);
  }

  renderer.setScissorTest(false);
}
