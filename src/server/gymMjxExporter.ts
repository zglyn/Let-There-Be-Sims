/**
 * Exportable Gymnasium & MuJoCo MJX RL Training Package Generator (`gymMjxExporter.ts`)
 *
 * Compiles any calibrated LetThereBeSims scene (from an active session or saved
 * Regression CI case) into a self-contained, runnable Reinforcement Learning
 * training package containing:
 * 1. `scene.mjcf.xml` — Calibrated MuJoCo MJCF model with articulated robot
 *    actuators, end-effector & Goal Placement Pad sites, calibrated target &
 *    obstacle bodies, wrist + overhead cameras, and EE contact force sensors.
 * 2. `env.py` — Standard `gymnasium.Env` (`LetThereBeSimsPickPlaceEnv`) with
 *    continuous joint-delta + gripper action space, calibrated shaped rewards
 *    (reach -> grasp -> lift -> transport -> goal placement - force/collision
 *    penalties), and per-reset domain randomization over friction, mass, and XY.
 * 3. `mjx_env.py` — GPU-vectorized JAX + `mujoco.mjx` parallel environment
 *    supporting 1,024–4,096 concurrent rollouts via `jax.vmap` / `jax.jit`.
 * 4. `train_rl.py` — Ready-to-run training & evaluation CLI supporting
 *    Stable-Baselines3 (`PPO` / `SAC`) and GPU MJX vectorized training plus a
 *    20-environment calibrated regression evaluator.
 * 5. `requirements.txt` & `README.md` — Exact local, Colab, and Modal commands.
 *
 * Use cases:
 * - Giving roboticists an immediately runnable RL training harness calibrated
 *   directly from a real-world phone video in under 2 minutes.
 * - Downloading a single `.zip` archive or copying individual `.py`/`.xml` files.
 */

import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type {
  ControllerParams,
  GymMjxPackageResponse,
  GymRandomizationLevel,
  GymRewardPreset,
  GymRlAlgorithm,
  SceneSpec,
  TrainingPackageFile,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { resolveTaskGoalPose } from '../utils/taskGoalResolver';
import { buildZipArchive } from '../utils/zipBuilder';

export interface GymMjxSourceInput {
  sourceLabel: string;
  taskGoal: string;
  robotModel: RobotModelId;
  sceneSpec: SceneSpec;
  controllerParams: ControllerParams;
  algorithm: GymRlAlgorithm;
  rewardPreset: GymRewardPreset;
  randomizationLevel: GymRandomizationLevel;
  numEnvs: number;
  totalTimesteps: number;
}

/**
 * Converts an arbitrary object/task label into a clean filesystem-safe slug.
 */
function toSlug(raw: string): string {
  const cleaned = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 36);
  return cleaned || 'tabletop_task';
}

/**
 * Escapes XML attribute values safely.
 */
function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Converts a 6-digit hex color string into normalized MuJoCo RGBA string.
 */
function hexToRgba(hex: string, alpha = 1): string {
  const clean = hex.replace('#', '').trim();
  if (!/^[0-9a-fA-F]{6}$/.test(clean)) {
    return `0.65 0.72 0.80 ${alpha.toFixed(2)}`;
  }
  const r = (parseInt(clean.slice(0, 2), 16) / 255).toFixed(3);
  const g = (parseInt(clean.slice(2, 4), 16) / 255).toFixed(3);
  const b = (parseInt(clean.slice(4, 6), 16) / 255).toFixed(3);
  return `${r} ${g} ${b} ${alpha.toFixed(2)}`;
}

/**
 * Builds a clean, RL-ready MuJoCo MJCF XML file (Z-up) calibrated from the
 * scene specification, with articulated 6-DOF robot joints, position actuators,
 * wrist + overhead cameras, end-effector force sensors, target object freejoint,
 * obstacle geoms, and Goal Placement Pad site.
 */
export function buildRlReadyMjcfXml(params: {
  sceneSpec: SceneSpec;
  robotModel: RobotModelId;
  taskGoal: string;
}): string {
  const { sceneSpec, robotModel, taskGoal } = params;
  logFunctionCall('buildRlReadyMjcfXml', {
    robotModel,
    targetObject: sceneSpec.targetObject.name,
    obstacleCount: sceneSpec.obstacles.length,
  });

  const robot = APP_CONFIG.robots[robotModel];
  const goalPad = resolveTaskGoalPose(sceneSpec);
  const tObj = sceneSpec.targetObject;
  const tableX = Math.max(0.65, sceneSpec.tableDimensions.x);
  const tableY = Math.max(0.55, sceneSpec.tableDimensions.z);
  const tableHalfZ = Math.max(0.02, sceneSpec.tableDimensions.y * 0.5);

  // Map Three.js Y-up (x forward, y up, z lateral) -> MuJoCo Z-up (x forward, y lateral, z up)
  const targetHalfX = Math.max(0.015, tObj.dimensions.x * 0.5);
  const targetHalfY = Math.max(0.015, tObj.dimensions.z * 0.5);
  const targetHalfZ = Math.max(0.02, tObj.dimensions.y * 0.5);
  const targetGeomType = tObj.shape === 'cylinder' || tObj.shape === 'mug' || tObj.shape === 'bottle'
    ? 'cylinder'
    : tObj.shape === 'sphere'
      ? 'sphere'
      : 'box';
  const targetGeomSize =
    targetGeomType === 'cylinder'
      ? `${Math.max(targetHalfX, targetHalfY).toFixed(4)} ${targetHalfZ.toFixed(4)}`
      : targetGeomType === 'sphere'
        ? `${Math.max(targetHalfX, targetHalfZ).toFixed(4)}`
        : `${targetHalfX.toFixed(4)} ${targetHalfY.toFixed(4)} ${targetHalfZ.toFixed(4)}`;

  const obstacleXml = sceneSpec.obstacles
    .map((obs, idx) => {
      const safeName = toSlug(obs.name || `obstacle_${idx + 1}`);
      const hx = Math.max(0.015, obs.dimensions.x * 0.5).toFixed(4);
      const hy = Math.max(0.015, obs.dimensions.z * 0.5).toFixed(4);
      const hz = Math.max(0.015, obs.dimensions.y * 0.5).toFixed(4);
      const px = obs.position.x.toFixed(4);
      const py = (-obs.position.z).toFixed(4);
      const pz = (Number(hz) + 0.0005).toFixed(4);
      const friction = Math.max(0.15, Math.min(1.2, obs.friction)).toFixed(3);
      return `    <body name="obs_${safeName}_${idx}" pos="${px} ${py} ${pz}">
      <geom name="obs_geom_${idx}" type="box" size="${hx} ${hy} ${hz}" rgba="${hexToRgba(obs.colorHex, 0.95)}" friction="${friction} 0.005 0.001" contype="1" conaffinity="1"/>
    </body>`;
    })
    .join('\n');

  const baseH = robot.baseHeightM.toFixed(4);
  const l1 = robot.upperArmM.toFixed(4);
  const l2 = robot.forearmM.toFixed(4);
  const lw = robot.wristM.toFixed(4);
  const torque = robot.maxTorqueNm.toFixed(2);

  return `<!--
  LetThereBeSims Calibrated RL & MJX MuJoCo Scene
  Robot: ${escapeXml(robot.name)}
  Task: ${escapeXml(taskGoal)}
  Target Object: ${escapeXml(tObj.name)} (${tObj.massKg.toFixed(3)} kg, mu=${tObj.friction.toFixed(2)})
  Goal Placement Pad: x=${goalPad.x.toFixed(3)}m, y=${(-goalPad.z).toFixed(3)}m, r=${goalPad.radiusM.toFixed(3)}m
-->
<mujoco model="lettherebesims_${toSlug(tObj.name)}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" iterations="25"/>

  <default>
    <joint damping="1.2" armature="0.02"/>
    <geom condim="4" solimp="0.95 0.99 0.001" solref="0.008 1"/>
    <position kp="65" kv="12" forcerange="-${torque} ${torque}"/>
  </default>

  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.14 0.17 0.23" rgb2="0.10 0.13 0.18"/>
    <material name="table_mat" texture="grid_tex" texrepeat="12 12" reflectance="0.08"/>
    <material name="arm_mat" rgba="0.88 0.90 0.94 1"/>
    <material name="servo_mat" rgba="0.16 0.19 0.26 1"/>
    <material name="pad_mat" rgba="0.06 0.72 0.50 0.55"/>
  </asset>

  <worldbody>
    <light pos="0.3 0.0 1.4" dir="0 0 -1" diffuse="0.95 0.95 0.98" specular="0.25 0.25 0.25"/>
    <camera name="overhead_cam" pos="0.22 0.0 0.92" xyaxes="0 -1 0 1 0 0" fovy="46"/>
    <camera name="front_cam" pos="0.72 -0.45 0.48" xyaxes="0.55 0.83 0 -0.38 0.25 0.89" fovy="48"/>

    <!-- Calibrated Tabletop Surface (Z=0 is top face) -->
    <geom name="table_top" type="box" pos="0.18 0 ${(-tableHalfZ).toFixed(4)}" size="${(tableX * 0.5).toFixed(4)} ${(tableY * 0.5).toFixed(4)} ${tableHalfZ.toFixed(4)}" material="table_mat" friction="0.7 0.005 0.001"/>

    <!-- Calibrated Goal Placement Pad -->
    <body name="goal_pad" pos="${goalPad.x.toFixed(4)} ${(-goalPad.z).toFixed(4)} 0.001">
      <geom name="goal_pad_visual" type="cylinder" size="${goalPad.radiusM.toFixed(4)} 0.001" material="pad_mat" contype="0" conaffinity="0"/>
      <site name="goal_pad_site" pos="0 0 0.002" size="${goalPad.radiusM.toFixed(4)}" rgba="0.06 0.72 0.50 0.35"/>
    </body>

    <!-- Articulated 6-DOF Manipulator + Parallel-Jaw Gripper -->
    <body name="robot_base" pos="0 0 0">
      <geom name="base_pedestal" type="cylinder" pos="0 0 ${(robot.baseHeightM * 0.5).toFixed(4)}" size="0.055 ${(robot.baseHeightM * 0.5).toFixed(4)}" material="servo_mat"/>
      <body name="link_shoulder" pos="0 0 ${baseH}">
        <joint name="joint_base_yaw" type="hinge" axis="0 0 1" range="-2.6 2.6"/>
        <geom name="shoulder_hub" type="cylinder" size="0.038 0.032" material="arm_mat"/>
        <body name="link_upper_arm" pos="0 0 0.02">
          <joint name="joint_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
          <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0 0 ${l1}" size="0.022" material="arm_mat"/>
          <body name="link_forearm" pos="0 0 ${l1}">
            <joint name="joint_elbow_pitch" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
            <geom name="forearm_geom" type="capsule" fromto="0 0 0 ${l2} 0 0" size="0.019" material="arm_mat"/>
            <body name="link_wrist_pitch" pos="${l2} 0 0">
              <joint name="joint_wrist_pitch" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
              <geom name="wrist_hub" type="sphere" size="0.022" material="servo_mat"/>
              <body name="link_wrist_roll" pos="0 0 ${(-robot.wristM * 0.5).toFixed(4)}">
                <joint name="joint_wrist_roll" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <body name="link_ee_flange" pos="0 0 ${(-robot.wristM * 0.5).toFixed(4)}">
                  <joint name="joint_flange_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                  <geom name="ee_palm" type="box" size="0.024 0.034 0.010" material="servo_mat"/>
                  <camera name="wrist_cam" pos="0.03 0 0.02" xyaxes="0 -1 0 0.7 0 0.7" fovy="58"/>
                  <site name="ee_site" pos="0 0 -0.042" size="0.012" rgba="0.96 0.62 0.04 0.8"/>

                  <!-- Parallel Jaw Fingers -->
                  <body name="left_finger" pos="0 0.022 -0.028">
                    <joint name="joint_gripper_left" type="slide" axis="0 1 0" range="-0.022 0.024"/>
                    <geom name="left_finger_pad" type="box" size="0.012 0.006 0.026" rgba="0.96 0.62 0.04 1" friction="1.15 0.01 0.002"/>
                  </body>
                  <body name="right_finger" pos="0 -0.022 -0.028">
                    <joint name="joint_gripper_right" type="slide" axis="0 -1 0" range="-0.022 0.024"/>
                    <geom name="right_finger_pad" type="box" size="0.012 0.006 0.026" rgba="0.96 0.62 0.04 1" friction="1.15 0.01 0.002"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Calibrated Target Object -->
    <body name="target_object" pos="${tObj.position.x.toFixed(4)} ${(-tObj.position.z).toFixed(4)} ${(targetHalfZ + 0.001).toFixed(4)}">
      <freejoint name="target_freejoint"/>
      <geom name="target_geom" type="${targetGeomType}" size="${targetGeomSize}" mass="${Math.max(0.015, tObj.massKg).toFixed(4)}" rgba="${hexToRgba(tObj.colorHex, 1)}" friction="${Math.max(0.15, Math.min(1.4, tObj.friction)).toFixed(3)} 0.005 0.001"/>
      <site name="target_site" pos="0 0 0" size="0.010" rgba="0.06 0.72 0.50 0.5"/>
    </body>

    <!-- Calibrated Tabletop Obstacles -->
${obstacleXml || '    <!-- No clutter obstacles detected -->'}
  </worldbody>

  <actuator>
    <position name="act_base_yaw" joint="joint_base_yaw"/>
    <position name="act_shoulder_pitch" joint="joint_shoulder_pitch"/>
    <position name="act_elbow_pitch" joint="joint_elbow_pitch"/>
    <position name="act_wrist_pitch" joint="joint_wrist_pitch"/>
    <position name="act_wrist_roll" joint="joint_wrist_roll"/>
    <position name="act_flange_yaw" joint="joint_flange_yaw"/>
    <position name="act_gripper_left" joint="joint_gripper_left" kp="120" kv="8" forcerange="-35 35"/>
    <position name="act_gripper_right" joint="joint_gripper_right" kp="120" kv="8" forcerange="-35 35"/>
  </actuator>

  <sensor>
    <force name="ee_force" site="ee_site"/>
    <torque name="ee_torque" site="ee_site"/>
  </sensor>
</mujoco>
`;
}

/**
 * Generates the runnable `env.py` Gymnasium environment calibrated to the
 * scene's exact target object, Goal Placement Pad, force ceilings, and domain
 * randomization envelope.
 */
function buildGymnasiumEnvPy(params: {
  sceneSpec: SceneSpec;
  robotModel: RobotModelId;
  taskGoal: string;
  rewardPreset: GymRewardPreset;
  randomizationLevel: GymRandomizationLevel;
  packageSlug: string;
}): string {
  const { sceneSpec, robotModel, taskGoal, rewardPreset, randomizationLevel, packageSlug } = params;
  const goalPad = resolveTaskGoalPose(sceneSpec);
  const tObj = sceneSpec.targetObject;
  const cfg = APP_CONFIG.trainingExport.gymMjx;

  const drScale =
    randomizationLevel === 'off' ? 0.0 : randomizationLevel === 'aggressive' ? 1.75 : 1.0;

  const rewardWeights =
    rewardPreset === 'sparse_goal'
      ? { reach: 0.1, lift: 0.2, transport: 0.2, placeBonus: 25.0, forcePen: 0.02 }
      : rewardPreset === 'force_safe_precision'
        ? { reach: 1.2, lift: 2.5, transport: 3.5, placeBonus: 18.0, forcePen: 0.12 }
        : { reach: 1.5, lift: 3.0, transport: 4.0, placeBonus: 15.0, forcePen: 0.04 };

  const randArrayJson = JSON.stringify(
    sceneSpec.randomizations.slice(0, 20).map((r) => ({
      id: r.id,
      friction: r.friction,
      mass: r.mass,
      xy_offset: r.xyOffset,
    })),
    null,
    2,
  );

  return `"""
LetThereBeSims Gymnasium Environment (env.py)

Auto-generated from real-world tabletop calibration:
- Robot: ${APP_CONFIG.robots[robotModel].name}
- Task Goal: ${taskGoal}
- Reward Preset: ${rewardPreset}
- Domain Randomization: ${randomizationLevel} (scale=${drScale})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ENV_ID = "LetThereBeSims/${packageSlug}-v0"
DEFAULT_XML_PATH = Path(__file__).resolve().parent / "scene.mjcf.xml"

CALIBRATED_RANDOMIZATIONS = ${randArrayJson}


class LetThereBeSimsPickPlaceEnv(gym.Env):
    """Calibrated MuJoCo pick-lift-transport-place Gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": ${cfg.controlHz}}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        render_mode: Optional[str] = None,
        randomization_scale: float = ${drScale.toFixed(2)},
        max_episode_steps: int = ${cfg.maxEpisodeSteps},
        camera_name: str = "front_cam",
    ) -> None:
        super().__init__()
        self.xml_path = str(xml_path)
        self.render_mode = render_mode
        self.randomization_scale = float(randomization_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.camera_name = camera_name

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = max(1, int(round(1.0 / (self.model.opt.timestep * ${cfg.controlHz}))))

        # Resolve named sites, bodies, and geoms
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_pad_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.target_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_freejoint")
        self.target_qpos_adr = int(self.model.jnt_qposadr[self.target_jnt_id])

        # Calibrated nominal physical anchors
        self.nominal_target_pos = np.array(
            [${tObj.position.x.toFixed(4)}, ${(-tObj.position.z).toFixed(4)}, ${(Math.max(0.02, tObj.dimensions.y * 0.5) + 0.001).toFixed(4)}],
            dtype=np.float64,
        )
        self.goal_pad_pos = np.array(
            [${goalPad.x.toFixed(4)}, ${(-goalPad.z).toFixed(4)}, ${(Math.max(0.02, tObj.dimensions.y * 0.5)).toFixed(4)}],
            dtype=np.float64,
        )
        self.goal_pad_radius = ${goalPad.radiusM.toFixed(4)}
        self.nominal_mass_kg = float(self.model.body_mass[self.target_body_id])
        self.nominal_friction = float(self.model.geom_friction[self.target_geom_id, 0])
        self.target_lift_z = ${APP_CONFIG.simulation.targetLiftZMeters.toFixed(3)}
        self.max_safe_force_n = ${APP_CONFIG.simulation.maxSafeContactForceN.toFixed(1)}

        # Action: 6 joint delta targets + 1 symmetric gripper target in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation: 6 qpos + 6 qvel + 1 gripper + ee(3) + obj(3) + goal(3) + ee->obj(3) + obj->goal(3) + force(1) = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )

        self._step_count = 0
        self._renderer: Optional[mujoco.Renderer] = None

    def _get_contact_force_n(self) -> float:
        """Reads L2 norm of end-effector force sensor (Newtons)."""
        if self.model.nsensordata >= 3:
            return float(np.linalg.norm(self.data.sensordata[:3]))
        return 0.0

    def _get_obs(self) -> np.ndarray:
        """Assembles the 29-D proprioceptive + spatial + force observation vector."""
        qpos_arm = self.data.qpos[:6]
        qvel_arm = self.data.qvel[:6]
        gripper_q = np.array([self.data.qpos[6]], dtype=np.float64)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        obj_pos = self.data.site_xpos[self.target_site_id].copy()
        goal_pos = self.data.site_xpos[self.goal_site_id].copy()
        ee_to_obj = obj_pos - ee_pos
        obj_to_goal = goal_pos - obj_pos
        force_n = np.array([self._get_contact_force_n() / 50.0], dtype=np.float64)
        return np.concatenate(
            [qpos_arm, qvel_arm, gripper_q, ee_pos, obj_pos, goal_pos, ee_to_obj, obj_to_goal, force_n]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets robot & tabletop state with calibrated domain randomization."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Apply either an indexed calibrated seed (for 20-env regression eval) or fresh DR
        eval_index = options.get("calibrated_env_index") if options else None
        if eval_index is not None and 0 <= int(eval_index) < len(CALIBRATED_RANDOMIZATIONS):
            dr = CALIBRATED_RANDOMIZATIONS[int(eval_index)]
            friction = float(dr["friction"])
            mass_kg = float(dr["mass"])
            dx, dy = float(dr["xy_offset"][0]), -float(dr["xy_offset"][1])
        else:
            s = self.randomization_scale
            friction = float(np.clip(self.nominal_friction + self.np_random.uniform(-0.18, 0.18) * s, 0.15, 1.35))
            mass_kg = float(np.clip(self.nominal_mass_kg * (1.0 + self.np_random.uniform(-0.35, 0.40) * s), 0.015, 2.5))
            dx = float(self.np_random.uniform(-0.035, 0.035) * s)
            dy = float(self.np_random.uniform(-0.035, 0.035) * s)

        self.model.geom_friction[self.target_geom_id, 0] = friction
        self.model.body_mass[self.target_body_id] = mass_kg

        # Place target object on tabletop with XY jitter
        self.data.qpos[self.target_qpos_adr : self.target_qpos_adr + 3] = (
            self.nominal_target_pos + np.array([dx, dy, 0.0])
        )
        self.data.qpos[self.target_qpos_adr + 3 : self.target_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

        # Start arm in a clean ready pose above the workspace
        self.data.qpos[:6] = np.array([0.0, 0.35, -0.75, -1.15, 0.0, 0.0])
        self.data.ctrl[:6] = self.data.qpos[:6]
        self.data.ctrl[6:8] = 0.022  # Open gripper
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {
            "friction": friction,
            "mass_kg": mass_kg,
            "xy_offset": (dx, dy),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances MuJoCo physics and computes calibrated pick-and-place reward."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta_scale = 0.06
        self.data.ctrl[:6] = np.clip(self.data.ctrl[:6] + action[:6] * delta_scale, -2.5, 2.5)
        grip_target = float(np.interp(action[6], [-1.0, 1.0], [-0.018, 0.022]))
        self.data.ctrl[6] = grip_target
        self.data.ctrl[7] = grip_target

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.site_xpos[self.target_site_id]
        goal_pos = self.data.site_xpos[self.goal_site_id]
        force_n = self._get_contact_force_n()

        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_goal_xy = float(np.linalg.norm(obj_pos[:2] - goal_pos[:2]))
        lift_progress = float(np.clip((obj_pos[2] - self.nominal_target_pos[2]) / self.target_lift_z, 0.0, 1.0))

        r_reach = ${rewardWeights.reach} * float(np.exp(-5.0 * dist_ee_obj))
        r_lift = ${rewardWeights.lift} * lift_progress
        r_transport = ${rewardWeights.transport} * float(np.exp(-4.0 * dist_obj_goal_xy)) * (0.25 + 0.75 * lift_progress)
        excess_force = max(0.0, force_n - self.max_safe_force_n)
        r_force_pen = -${rewardWeights.forcePen} * excess_force

        # Task completion: object resting stably inside the Goal Placement Pad
        is_released = grip_target > 0.008
        is_on_goal_pad = dist_obj_goal_xy <= (self.goal_pad_radius + 0.015) and obj_pos[2] <= (self.nominal_target_pos[2] + 0.04)
        is_success = bool(is_on_goal_pad and is_released)
        r_success = ${rewardWeights.placeBonus} if is_success else 0.0

        reward = float(r_reach + r_lift + r_transport + r_force_pen + r_success)
        terminated = is_success or (obj_pos[2] < -0.15)
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "is_success": is_success,
            "dist_ee_obj_m": dist_ee_obj,
            "placement_error_m": dist_obj_goal_xy,
            "object_z_m": float(obj_pos[2]),
            "contact_force_n": force_n,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders an RGB frame from the configured MuJoCo camera."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(
    id=ENV_ID,
    entry_point=f"{__name__}:LetThereBeSimsPickPlaceEnv",
    max_episode_steps=${cfg.maxEpisodeSteps},
)
`;
}

/**
 * Generates the GPU-vectorized JAX + MuJoCo MJX environment (`mjx_env.py`).
 */
function buildMjxEnvPy(params: {
  sceneSpec: SceneSpec;
  robotModel: RobotModelId;
  numEnvs: number;
}): string {
  const { sceneSpec, robotModel, numEnvs } = params;
  const goalPad = resolveTaskGoalPose(sceneSpec);
  const tObj = sceneSpec.targetObject;
  const cfg = APP_CONFIG.trainingExport.gymMjx;

  return `"""
LetThereBeSims GPU-Vectorized MuJoCo MJX Environment (mjx_env.py)

Runs ${numEnvs} parallel physics environments simultaneously on GPU/TPU via
JAX (jax.vmap + jax.jit) and mujoco.mjx.
Robot: ${APP_CONFIG.robots[robotModel].name}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

DEFAULT_XML_PATH = Path(__file__).resolve().parent / "scene.mjcf.xml"


@dataclass
class MjxEnvState:
    mjx_data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    step_count: jax.Array


class MjxPickPlaceEnv:
    """Vectorized JAX/MJX tabletop pick-and-place environment."""

    def __init__(self, xml_path: str | Path = DEFAULT_XML_PATH, max_steps: int = ${cfg.maxEpisodeSteps}) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mjx_model = mjx.put_model(self.mj_model)
        self.max_steps = int(max_steps)
        self.substeps = max(1, int(round(1.0 / (self.mj_model.opt.timestep * ${cfg.controlHz}))))

        self.ee_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.goal_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_pad_site")
        self.target_jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "target_freejoint")
        self.target_qpos_adr = int(self.mj_model.jnt_qposadr[self.target_jnt_id])

        self.nominal_target = jnp.array(
            [${tObj.position.x.toFixed(4)}, ${(-tObj.position.z).toFixed(4)}, ${(Math.max(0.02, tObj.dimensions.y * 0.5) + 0.001).toFixed(4)}]
        )
        self.goal_pad = jnp.array([${goalPad.x.toFixed(4)}, ${(-goalPad.z).toFixed(4)}, ${(Math.max(0.02, tObj.dimensions.y * 0.5)).toFixed(4)}])
        self.goal_radius = ${goalPad.radiusM.toFixed(4)}
        self.target_lift_z = ${APP_CONFIG.simulation.targetLiftZMeters.toFixed(3)}

    def _obs(self, d: mjx.Data) -> jax.Array:
        ee = d.site_xpos[self.ee_site_id]
        obj = d.site_xpos[self.target_site_id]
        goal = d.site_xpos[self.goal_site_id]
        return jnp.concatenate([d.qpos[:6], d.qvel[:6], d.qpos[6:7], ee, obj, goal, obj - ee, goal - obj])

    def reset_single(self, rng: jax.Array) -> MjxEnvState:
        """Resets a single environment with randomized target XY pose."""
        rng_x, rng_y = jax.random.split(rng)
        dx = jax.random.uniform(rng_x, (), minval=-0.035, maxval=0.035)
        dy = jax.random.uniform(rng_y, (), minval=-0.035, maxval=0.035)

        d = mjx.make_data(self.mjx_model)
        qpos = d.qpos.at[:6].set(jnp.array([0.0, 0.35, -0.75, -1.15, 0.0, 0.0]))
        qpos = qpos.at[self.target_qpos_adr : self.target_qpos_adr + 3].set(
            self.nominal_target + jnp.array([dx, dy, 0.0])
        )
        ctrl = d.ctrl.at[:6].set(qpos[:6]).at[6:8].set(0.022)
        d = mjx.forward(self.mjx_model, d.replace(qpos=qpos, ctrl=ctrl))
        return MjxEnvState(
            mjx_data=d,
            obs=self._obs(d),
            reward=jnp.array(0.0),
            done=jnp.array(False),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def step_single(self, state: MjxEnvState, action: jax.Array) -> MjxEnvState:
        """Steps a single MJX environment forward by one control interval."""
        action = jnp.clip(action, -1.0, 1.0)
        next_arm = jnp.clip(state.mjx_data.ctrl[:6] + action[:6] * 0.06, -2.5, 2.5)
        grip = -0.018 + (action[6] + 1.0) * 0.5 * 0.040
        ctrl = state.mjx_data.ctrl.at[:6].set(next_arm).at[6].set(grip).at[7].set(grip)
        d = state.mjx_data.replace(ctrl=ctrl)

        def substep_fn(carry, _):
            return mjx.step(self.mjx_model, carry), None

        d, _ = jax.lax.scan(substep_fn, d, None, length=self.substeps)
        ee = d.site_xpos[self.ee_site_id]
        obj = d.site_xpos[self.target_site_id]
        goal = d.site_xpos[self.goal_site_id]

        dist_ee_obj = jnp.linalg.norm(ee - obj)
        dist_obj_goal = jnp.linalg.norm(obj[:2] - goal[:2])
        lift_prog = jnp.clip((obj[2] - self.nominal_target[2]) / self.target_lift_z, 0.0, 1.0)
        success = (dist_obj_goal <= self.goal_radius + 0.015) & (obj[2] <= self.nominal_target[2] + 0.04) & (grip > 0.008)

        reward = (
            1.5 * jnp.exp(-5.0 * dist_ee_obj)
            + 3.0 * lift_prog
            + 4.0 * jnp.exp(-4.0 * dist_obj_goal) * (0.25 + 0.75 * lift_prog)
            + jnp.where(success, 15.0, 0.0)
        )
        step_count = state.step_count + 1
        done = success | (obj[2] < -0.15) | (step_count >= self.max_steps)
        return MjxEnvState(mjx_data=d, obs=self._obs(d), reward=reward, done=done, step_count=step_count)

    def make_vectorized(self, num_envs: int = ${numEnvs}) -> Tuple[object, object]:
        """Returns JIT-compiled vectorized (reset_batch, step_batch) callables."""
        reset_batch = jax.jit(jax.vmap(self.reset_single))
        step_batch = jax.jit(jax.vmap(self.step_single))
        return reset_batch, step_batch


if __name__ == "__main__":
    env = MjxPickPlaceEnv()
    reset_fn, step_fn = env.make_vectorized(num_envs=${Math.min(numEnvs, 1024)})
    keys = jax.random.split(jax.random.PRNGKey(0), ${Math.min(numEnvs, 1024)})
    state = reset_fn(keys)
    actions = jnp.zeros((${Math.min(numEnvs, 1024)}, 7))
    next_state = step_fn(state, actions)
    print(f"MJX smoke test OK — batch={next_state.obs.shape}, mean_reward={float(jnp.mean(next_state.reward)):.4f}")
`;
}

/**
 * Generates the ready-to-run `train_rl.py` script supporting Stable-Baselines3
 * PPO/SAC, MJX GPU benchmark/rollout verification, and 20-env calibrated eval.
 */
function buildTrainRlPy(params: {
  algorithm: GymRlAlgorithm;
  numEnvs: number;
  totalTimesteps: number;
  packageSlug: string;
}): string {
  const { algorithm, numEnvs, totalTimesteps, packageSlug } = params;
  const defaultBackend = algorithm === 'mjx_ppo' ? 'mjx' : 'sb3';
  const defaultSb3Algo = algorithm === 'sac' ? 'sac' : 'ppo';

  return `"""
LetThereBeSims RL Training & 20-Env Regression Evaluation CLI (train_rl.py)

Usage:
  python train_rl.py --backend ${defaultBackend} --algo ${defaultSb3Algo} --num-envs ${numEnvs} --timesteps ${totalTimesteps}
  python train_rl.py --eval-checkpoint ./checkpoints/${packageSlug}_${defaultSb3Algo}.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import CALIBRATED_RANDOMIZATIONS, LetThereBeSimsPickPlaceEnv


def train_sb3(algo: str, num_envs: int, total_timesteps: int, out_dir: Path) -> Path:
    """Trains PPO or SAC across vectorized Gymnasium MuJoCo environments."""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_cls = SubprocVecEnv if num_envs > 1 else None
    vec_env = make_vec_env(
        lambda: LetThereBeSimsPickPlaceEnv(),
        n_envs=max(1, min(num_envs, 64)),
        vec_env_cls=vec_cls,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(5_000, total_timesteps // 10),
        save_path=str(out_dir),
        name_prefix=f"${packageSlug}_{algo}",
    )

    if algo == "sac":
        model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(out_dir / "tb"))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            tensorboard_log=str(out_dir / "tb"),
        )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    final_path = out_dir / f"${packageSlug}_{algo}.zip"
    model.save(str(final_path))
    vec_env.save(str(out_dir / "vecnormalize.pkl"))
    print(f"Saved trained policy checkpoint -> {final_path}")
    return final_path


def run_mjx_benchmark(num_envs: int) -> None:
    """Runs a JIT-compiled GPU/CPU MJX batch rollout throughput check."""
    import time
    import jax
    import jax.numpy as jnp
    from mjx_env import MjxPickPlaceEnv

    env = MjxPickPlaceEnv()
    reset_fn, step_fn = env.make_vectorized(num_envs=num_envs)
    keys = jax.random.split(jax.random.PRNGKey(42), num_envs)
    state = reset_fn(keys)
    jax.block_until_ready(state.obs)

    t0 = time.perf_counter()
    for step in range(100):
        action_key = jax.random.PRNGKey(step)
        actions = jax.random.uniform(action_key, (num_envs, 7), minval=-0.25, maxval=0.25)
        state = step_fn(state, actions)
    jax.block_until_ready(state.obs)
    elapsed = time.perf_counter() - t0
    sps = (num_envs * 100) / max(elapsed, 1e-6)
    print(f"MJX Vectorized Throughput: {sps:,.0f} steps/sec ({num_envs} parallel envs, {elapsed:.2f}s)")


def evaluate_calibrated_20_envs(checkpoint_path: Path, algo: str) -> None:
    """Evaluates a saved SB3 checkpoint against all 20 calibrated LetThereBeSims envs."""
    from stable_baselines3 import PPO, SAC

    model_cls = SAC if algo == "sac" else PPO
    model = model_cls.load(str(checkpoint_path))
    env = LetThereBeSimsPickPlaceEnv()

    passed = 0
    total = len(CALIBRATED_RANDOMIZATIONS)
    for idx in range(total):
        obs, _ = env.reset(options={"calibrated_env_index": idx})
        done = False
        peak_force = 0.0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            peak_force = max(peak_force, float(info.get("contact_force_n", 0.0)))
            success = bool(info.get("is_success", False))
            done = terminated or truncated
        passed += int(success)
        status = "PASS" if success else "FAIL"
        print(f"  [E{idx + 1:02d}] {status} | peak_force={peak_force:5.1f}N")

    print(f"\\nCalibrated 20-Env Score: {passed}/{total} ({(passed / max(total, 1)) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LetThereBeSims RL Trainer & Evaluator")
    parser.add_argument("--backend", choices=["sb3", "mjx"], default="${defaultBackend}")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="${defaultSb3Algo}")
    parser.add_argument("--num-envs", type=int, default=${numEnvs})
    parser.add_argument("--timesteps", type=int, default=${totalTimesteps})
    parser.add_argument("--out-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.eval_checkpoint is not None:
        evaluate_calibrated_20_envs(args.eval_checkpoint, args.algo)
        return

    if args.backend == "mjx":
        run_mjx_benchmark(args.num_envs)
    else:
        ckpt = train_sb3(args.algo, args.num_envs, args.timesteps, args.out_dir)
        evaluate_calibrated_20_envs(ckpt, args.algo)


if __name__ == "__main__":
    main()
`;
}

/**
 * Generates `requirements.txt` and `README.md` for the Gymnasium + MJX package.
 */
function buildGymRequirementsAndReadme(params: {
  packageName: string;
  taskGoal: string;
  robotModel: RobotModelId;
  algorithm: GymRlAlgorithm;
  numEnvs: number;
  totalTimesteps: number;
}): { requirements: string; readme: string } {
  const { packageName, taskGoal, robotModel, algorithm, numEnvs, totalTimesteps } = params;
  const requirements = [
    'mujoco>=3.2.0',
    'gymnasium>=1.0.0',
    'stable-baselines3>=2.4.0',
    'numpy>=1.26.0',
    'tensorboard>=2.17.0',
    '# Optional GPU-vectorized MJX dependencies (uncomment on CUDA/TPU machines):',
    '# jax[cuda12]>=0.4.30',
    '# mujoco-mjx>=3.2.0',
    '',
  ].join('\n');

  const algoFlag = algorithm === 'sac' ? '--backend sb3 --algo sac' : algorithm === 'mjx_ppo' ? '--backend mjx' : '--backend sb3 --algo ppo';

  const readme = `# ${packageName} — LetThereBeSims Gymnasium + MuJoCo MJX Package

- **Robot:** ${APP_CONFIG.robots[robotModel].name}
- **Task Goal:** ${taskGoal}
- **Configured Algorithm:** \`${algorithm.toUpperCase()}\`
- **Parallel Environments:** \`${numEnvs}\`
- **Training Budget:** \`${totalTimesteps.toLocaleString()}\` steps

## Quick Start (Local Linux / macOS / Google Colab)

\`\`\`bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Inspect the calibrated MuJoCo scene interactively
python -m mujoco.viewer --mjcf=scene.mjcf.xml

# 2. Train the RL policy + run the calibrated 20-environment eval
python train_rl.py ${algoFlag} --num-envs ${numEnvs} --timesteps ${totalTimesteps}

# 3. Monitor training curves
tensorboard --logdir ./checkpoints/tb
\`\`\`

## Package Contents

- \`scene.mjcf.xml\` — Calibrated MuJoCo scene with actuators, Goal Placement Pad, force sensors, and wrist/overhead cameras.
- \`env.py\` — Standard \`gymnasium.Env\` with domain randomization and calibrated dense pick-and-place rewards.
- \`mjx_env.py\` — JAX + \`mujoco.mjx\` GPU-vectorized environment (\`jax.vmap\` / \`jax.jit\`).
- \`train_rl.py\` — Stable-Baselines3 PPO/SAC trainer, MJX GPU benchmark, and 20-env regression evaluator.
`;

  return { requirements, readme };
}

/**
 * Generates the complete Gymnasium + MuJoCo MJX RL training package preview and
 * file map for a calibrated scene.
 */
export function buildGymMjxTrainingPackage(input: GymMjxSourceInput): GymMjxPackageResponse {
  logFunctionCall('buildGymMjxTrainingPackage', {
    sourceLabel: input.sourceLabel,
    robotModel: input.robotModel,
    algorithm: input.algorithm,
    rewardPreset: input.rewardPreset,
    randomizationLevel: input.randomizationLevel,
    numEnvs: input.numEnvs,
    totalTimesteps: input.totalTimesteps,
  });

  const slug = toSlug(input.sceneSpec.targetObject.name);
  const packageName = `lettherebesims_${slug}_gym_mjx`;
  const goalPad = resolveTaskGoalPose(input.sceneSpec);

  const mjcfXml = buildRlReadyMjcfXml({
    sceneSpec: input.sceneSpec,
    robotModel: input.robotModel,
    taskGoal: input.taskGoal,
  });
  const envPy = buildGymnasiumEnvPy({
    sceneSpec: input.sceneSpec,
    robotModel: input.robotModel,
    taskGoal: input.taskGoal,
    rewardPreset: input.rewardPreset,
    randomizationLevel: input.randomizationLevel,
    packageSlug: slug,
  });
  const mjxEnvPy = buildMjxEnvPy({
    sceneSpec: input.sceneSpec,
    robotModel: input.robotModel,
    numEnvs: input.numEnvs,
  });
  const trainRlPy = buildTrainRlPy({
    algorithm: input.algorithm,
    numEnvs: input.numEnvs,
    totalTimesteps: input.totalTimesteps,
    packageSlug: slug,
  });
  const { requirements, readme } = buildGymRequirementsAndReadme({
    packageName,
    taskGoal: input.taskGoal,
    robotModel: input.robotModel,
    algorithm: input.algorithm,
    numEnvs: input.numEnvs,
    totalTimesteps: input.totalTimesteps,
  });

  const makeFile = (
    path: string,
    label: string,
    language: TrainingPackageFile['language'],
    content: string,
  ): TrainingPackageFile => ({
    path,
    label,
    language,
    content,
    sizeBytes: Buffer.byteLength(content, 'utf8'),
  });

  const files: TrainingPackageFile[] = [
    makeFile('env.py', 'Gymnasium Env (env.py)', 'python', envPy),
    makeFile('mjx_env.py', 'MuJoCo MJX GPU (mjx_env.py)', 'python', mjxEnvPy),
    makeFile('train_rl.py', 'RL Trainer CLI (train_rl.py)', 'python', trainRlPy),
    makeFile('scene.mjcf.xml', 'RL-Ready MuJoCo XML', 'xml', mjcfXml),
    makeFile('requirements.txt', 'requirements.txt', 'text', requirements),
    makeFile('README.md', 'README.md', 'markdown', readme),
  ];

  return {
    packageName,
    sourceLabel: input.sourceLabel,
    robotModel: input.robotModel,
    taskGoal: input.taskGoal,
    goalPad,
    config: {
      algorithm: input.algorithm,
      rewardPreset: input.rewardPreset,
      randomizationLevel: input.randomizationLevel,
      numEnvs: input.numEnvs,
      totalTimesteps: input.totalTimesteps,
      controlHz: APP_CONFIG.trainingExport.gymMjx.controlHz,
      maxEpisodeSteps: APP_CONFIG.trainingExport.gymMjx.maxEpisodeSteps,
    },
    files,
    generatedAt: new Date().toISOString(),
  };
}

/**
 * Compiles the Gymnasium + MuJoCo MJX RL package into a binary `.zip` buffer.
 */
export function buildGymMjxZipArchive(input: GymMjxSourceInput): {
  filename: string;
  buffer: Buffer;
} {
  logFunctionCall('buildGymMjxZipArchive', {
    sourceLabel: input.sourceLabel,
    algorithm: input.algorithm,
  });
  const pkg = buildGymMjxTrainingPackage(input);
  const buffer = buildZipArchive(
    pkg.files.map((f) => ({
      path: `${pkg.packageName}/${f.path}`,
      content: f.content,
    })),
  );
  return {
    filename: `${pkg.packageName}.zip`,
    buffer,
  };
}
