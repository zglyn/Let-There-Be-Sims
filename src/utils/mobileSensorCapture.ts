/**
 * Mobile DeviceOrientation Gravity Vector & Audio Level Utility (`mobileSensorCapture.ts`)
 *
 * Captures physical smartphone IMU telemetry and live microphone audio activity
 * during QR-paired judge hand-demonstration recording:
 * 1. Requests iOS 13+ `DeviceOrientationEvent` / `DeviceMotionEvent` permissions
 *    cleanly inside a user tap handler while remaining non-blocking on Android
 *    and desktop browsers.
 * 2. Computes the 3D gravity vector `(gx, gy, gz)` in `m/s^2` and camera horizon
 *    tilt angle (`cameraTiltDeg`) from live IMU events so Gemini 4 Carbon can
 *    level the tabletop plane (`y = 0`) accurately regardless of phone pitch/roll.
 * 3. Attaches a lightweight Web Audio API `AnalyserNode` to measure live mic
 *    RMS amplitude so the judge sees immediate confirmation that voice
 *    instructions are being recorded into the video's audio track.
 *
 * Use cases:
 * - Powering the live gravity-vector & microphone HUD on `/capture?session=<id>`.
 * - Averaging steady IMU readings across a 6-second hand demonstration clip.
 */

import { logFunctionCall } from './logger';

export interface LiveOrientationReading {
  gravityVector: { x: number; y: number; z: number };
  orientationDeg: {
    alpha: number;
    beta: number;
    gamma: number;
    cameraTiltDeg: number;
  };
  source: 'device_motion' | 'device_orientation' | 'fallback_level';
}

const GRAVITY_MS2 = 9.81;

/**
 * Computes a 3D gravity vector (m/s^2) and camera elevation tilt angle from
 * W3C DeviceOrientation Euler angles (`alpha`, `beta`, `gamma` in degrees).
 */
export function computeGravityFromEuler(
  alphaDeg: number,
  betaDeg: number,
  gammaDeg: number,
): LiveOrientationReading {
  const beta = (betaDeg * Math.PI) / 180;
  const gamma = (gammaDeg * Math.PI) / 180;

  // Device-frame gravity projection (Z points out of screen, Y up phone screen)
  const gx = Number((Math.sin(gamma) * Math.cos(beta) * GRAVITY_MS2).toFixed(3));
  const gy = Number((-Math.sin(beta) * GRAVITY_MS2).toFixed(3));
  const gz = Number((-Math.cos(gamma) * Math.cos(beta) * GRAVITY_MS2).toFixed(3));

  // Camera tilt below horizontal when filming a tabletop (0° = level forward, 90° = straight down)
  const cameraTiltDeg = Number(Math.max(-90, Math.min(90, 90 - betaDeg)).toFixed(1));

  return {
    gravityVector: { x: gx, y: gy, z: gz },
    orientationDeg: {
      alpha: Number(alphaDeg.toFixed(1)),
      beta: Number(betaDeg.toFixed(1)),
      gamma: Number(gammaDeg.toFixed(1)),
      cameraTiltDeg,
    },
    source: 'device_orientation',
  };
}

/**
 * Requests iOS 13+ IMU permissions when supported. Returns `true` if orientation
 * events are permitted or require no explicit permission prompt.
 */
export async function requestMobileOrientationPermission(): Promise<boolean> {
  logFunctionCall('requestMobileOrientationPermission');
  if (typeof window === 'undefined') return false;

  try {
    const oriCtor = window.DeviceOrientationEvent as unknown as {
      requestPermission?: () => Promise<'granted' | 'denied'>;
    };
    if (oriCtor && typeof oriCtor.requestPermission === 'function') {
      const res = await oriCtor.requestPermission();
      if (res !== 'granted') return false;
    }

    const motCtor = window.DeviceMotionEvent as unknown as {
      requestPermission?: () => Promise<'granted' | 'denied'>;
    };
    if (motCtor && typeof motCtor.requestPermission === 'function') {
      await motCtor.requestPermission().catch(() => 'denied');
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Subscribes to live `deviceorientation` and `devicemotion` events and emits
 * normalized gravity vector readings. Returns an unsubscribe cleanup callback.
 */
export function subscribeMobileGravitySensor(
  onReading: (reading: LiveOrientationReading) => void,
): () => void {
  logFunctionCall('subscribeMobileGravitySensor');
  if (typeof window === 'undefined') return () => {};

  let lastEuler = { alpha: 0, beta: 45, gamma: 0 };

  const handleOrientation = (ev: DeviceOrientationEvent): void => {
    if (typeof ev.beta !== 'number' || typeof ev.gamma !== 'number') return;
    const alpha = typeof ev.alpha === 'number' ? ev.alpha : 0;
    lastEuler = { alpha, beta: ev.beta, gamma: ev.gamma };
    onReading(computeGravityFromEuler(alpha, ev.beta, ev.gamma));
  };

  const handleMotion = (ev: DeviceMotionEvent): void => {
    const acc = ev.accelerationIncludingGravity;
    if (
      !acc ||
      typeof acc.x !== 'number' ||
      typeof acc.y !== 'number' ||
      typeof acc.z !== 'number'
    ) {
      return;
    }
    const mag = Math.hypot(acc.x, acc.y, acc.z);
    if (mag < 2.0) return;
    const norm = GRAVITY_MS2 / mag;
    const gx = Number((acc.x * norm).toFixed(3));
    const gy = Number((acc.y * norm).toFixed(3));
    const gz = Number((acc.z * norm).toFixed(3));
    const cameraTiltDeg = Number(
      Math.max(-90, Math.min(90, (Math.asin(Math.max(-1, Math.min(1, -gz / GRAVITY_MS2))) * 180) / Math.PI)).toFixed(1),
    );
    onReading({
      gravityVector: { x: gx, y: gy, z: gz },
      orientationDeg: {
        alpha: Number(lastEuler.alpha.toFixed(1)),
        beta: Number(lastEuler.beta.toFixed(1)),
        gamma: Number(lastEuler.gamma.toFixed(1)),
        cameraTiltDeg,
      },
      source: 'device_motion',
    });
  };

  window.addEventListener('deviceorientation', handleOrientation, true);
  window.addEventListener('devicemotion', handleMotion, true);

  return () => {
    window.removeEventListener('deviceorientation', handleOrientation, true);
    window.removeEventListener('devicemotion', handleMotion, true);
  };
}

/**
 * Starts a lightweight Web Audio RMS level meter for an active MediaStream
 * containing an audio track. Returns a cleanup function that closes the context.
 */
export function startAudioLevelMeter(
  stream: MediaStream,
  onLevel: (normalizedRms: number) => void,
): () => void {
  if (typeof window === 'undefined' || stream.getAudioTracks().length === 0) {
    onLevel(0);
    return () => {};
  }

  try {
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return () => {};

    const ctx = new AudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    let rafId = 0;

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sumSq = 0;
      for (let i = 0; i < data.length; i++) {
        const centered = (data[i] - 128) / 128;
        sumSq += centered * centered;
      }
      const rms = Math.sqrt(sumSq / data.length);
      onLevel(Math.min(1, rms * 4.5));
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(rafId);
      source.disconnect();
      void ctx.close().catch(() => {});
    };
  } catch {
    return () => {};
  }
}
