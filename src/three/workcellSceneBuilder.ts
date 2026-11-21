/**
 * Procedural 3D Robotics Workcell, Reconstructed Objects & Telemetry Overlays
 *
 * Builds the calibrated metric laboratory tabletop (`CanvasTexture` with
 * millimeter/decimeter calibration grid in `SRGBColorSpace` and max
 * anisotropy), soft contact shadow grounding, reconstructed target object
 * (ceramic mug with handle, bottle, cylinder, box, or sphere), obstacle
 * volumes, 3D end-effector trajectory spline, and dynamic contact-force vector
 * arrow.
 *
 * Use cases:
 * - Rendering Carbon's real-to-sim spatial calibration scene dynamically.
 * - Visualizing per-environment domain randomization offsets (mass, XY jitter)
 *   and live force spikes (`> 45 N`) in Precision Lab Dark telemetry colors.
 */

import * as THREE from 'three';
import type { SceneObjectSpec, TrajectoryFrame } from '../types/robotics';
import { logFunctionCall } from '../utils/logger';

/**
 * Generates a high-resolution procedural metric workbench grid texture on an
 * offscreen HTML5 canvas with explicit SRGBColorSpace and grazing-angle sharpness.
 */
export function createWorkcellGridTexture(maxAnisotropy: number): THREE.CanvasTexture {
  logFunctionCall('createWorkcellGridTexture', { maxAnisotropy });
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    ctx.fillStyle = '#0d131f';
    ctx.fillRect(0, 0, size, size);

    // Subtle minor grid (2.5 cm divisions)
    ctx.strokeStyle = 'rgba(51, 70, 99, 0.28)';
    ctx.lineWidth = 1;
    const minorStep = size / 32;
    for (let i = 0; i <= 32; i++) {
      const p = Math.round(i * minorStep) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(size, p);
      ctx.stroke();
    }

    // Major calibration grid (10 cm divisions)
    ctx.strokeStyle = 'rgba(100, 128, 168, 0.42)';
    ctx.lineWidth = 1.5;
    const majorStep = size / 8;
    for (let i = 0; i <= 8; i++) {
      const p = Math.round(i * majorStep) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(size, p);
      ctx.stroke();
    }

    // Crosshair calibration ticks at major intersections
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.55)';
    ctx.lineWidth = 1.5;
    for (let ix = 1; ix < 8; ix++) {
      for (let iy = 1; iy < 8; iy++) {
        const cx = ix * majorStep;
        const cy = iy * majorStep;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy);
        ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5);
        ctx.lineTo(cx, cy + 5);
        ctx.stroke();
      }
    }

    // Outer workcell frame rule
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.55)';
    ctx.lineWidth = 4;
    ctx.strokeRect(10, 10, size - 20, size - 20);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.max(1, maxAnisotropy);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates a soft radial contact shadow plane texture so grounded tabletop
 * assets sit flush with realistic ambient occlusion.
 */
export function createContactShadowTexture(): THREE.CanvasTexture {
  logFunctionCall('createContactShadowTexture');
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const grad = ctx.createRadialGradient(size / 2, size / 2, 12, size / 2, size / 2, size / 2);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.55)');
    grad.addColorStop(0.6, 'rgba(0, 0, 0, 0.18)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Constructs a composite procedural mesh (`THREE.Group`) for a reconstructed
 * tabletop object (mug, bottle, cylinder, box, or sphere) centered so its
 * geometric midpoint aligns cleanly with physics pose updates.
 */
export function buildSceneObjectMesh(spec: SceneObjectSpec, isTarget: boolean): THREE.Group {
  logFunctionCall('buildSceneObjectMesh', { name: spec.name, shape: spec.shape, isTarget });
  const group = new THREE.Group();
  group.name = `${isTarget ? 'target' : 'obstacle'}_${spec.name}`;

  const baseColor = new THREE.Color(spec.colorHex || (isTarget ? '#e2e8f0' : '#475569'));
  const mainMat = new THREE.MeshStandardMaterial({
    color: baseColor,
    roughness: spec.shape === 'bottle' ? 0.22 : isTarget ? 0.32 : 0.65,
    metalness: spec.shape === 'cylinder' ? 0.55 : 0.08,
  });

  const radius = Math.max(0.022, Math.min(spec.dimensions.x, spec.dimensions.z) * 0.5);
  const height = Math.max(0.045, spec.dimensions.y);

  if (spec.shape === 'mug') {
    // Composite ceramic mug: hollow-looking rimmed body + torus handle + inner coffee surface
    const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.92, height, 28), mainMat);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    const rimMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      roughness: 0.2,
      metalness: 0.05,
    });
    const rim = new THREE.Mesh(new THREE.TorusGeometry(radius * 0.96, 0.0035, 10, 28), rimMat);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = height * 0.5;
    group.add(rim);

    const handle = new THREE.Mesh(new THREE.TorusGeometry(height * 0.28, 0.0055, 12, 24, Math.PI), mainMat);
    handle.position.set(radius * 0.94, 0, 0);
    handle.rotation.z = -Math.PI / 2;
    handle.castShadow = true;
    group.add(handle);

    const interior = new THREE.Mesh(
      new THREE.CircleGeometry(radius * 0.88, 24),
      new THREE.MeshStandardMaterial({ color: 0x1e1b18, roughness: 0.18, metalness: 0.1 }),
    );
    interior.rotation.x = -Math.PI / 2;
    interior.position.y = height * 0.48;
    group.add(interior);
  } else if (spec.shape === 'bottle') {
    const bodyH = height * 0.68;
    const neckH = height * 0.22;
    const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.96, bodyH, 24), mainMat);
    body.position.y = -height * 0.5 + bodyH * 0.5;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    const shoulder = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.42, radius, height * 0.1, 24),
      mainMat,
    );
    shoulder.position.y = -height * 0.5 + bodyH + height * 0.05;
    shoulder.castShadow = true;
    group.add(shoulder);

    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.38, radius * 0.42, neckH, 20),
      mainMat,
    );
    neck.position.y = -height * 0.5 + bodyH + height * 0.1 + neckH * 0.5;
    group.add(neck);

    const cap = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.44, radius * 0.44, 0.014, 18),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.7, metalness: 0.1 }),
    );
    cap.position.y = height * 0.5 + 0.004;
    group.add(cap);
  } else if (spec.shape === 'sphere') {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 28, 20), mainMat);
    sphere.castShadow = true;
    sphere.receiveShadow = true;
    group.add(sphere);
  } else if (spec.shape === 'box') {
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(spec.dimensions.x, height, spec.dimensions.z),
      mainMat,
    );
    box.castShadow = true;
    box.receiveShadow = true;
    group.add(box);
  } else {
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 26), mainMat);
    cyl.castShadow = true;
    cyl.receiveShadow = true;
    group.add(cyl);
  }

  // Calibration reticle ring around target object base
  if (isTarget) {
    const reticle = new THREE.Mesh(
      new THREE.RingGeometry(radius + 0.012, radius + 0.016, 36),
      new THREE.MeshBasicMaterial({
        color: 0x06b6d4,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.72,
      }),
    );
    reticle.rotation.x = -Math.PI / 2;
    reticle.position.y = -height * 0.5 + 0.002;
    group.add(reticle);
  }

  return group;
}

/**
 * Builds a glowing calibration spline line representing the end-effector
 * trajectory across the selected rollout's 60 keyframes.
 */
export function buildTrajectoryLine(frames: TrajectoryFrame[], success: boolean): THREE.Line {
  logFunctionCall('buildTrajectoryLine', { frameCount: frames.length, success });
  const points = frames.map(
    (f) => new THREE.Vector3(f.endEffectorPos.x, f.endEffectorPos.y, f.endEffectorPos.z),
  );
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: success ? 0x06b6d4 : 0xf43f5e,
    transparent: true,
    opacity: 0.85,
  });
  return new THREE.Line(geometry, material);
}
