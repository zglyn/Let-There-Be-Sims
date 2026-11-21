/**
 * Synthetic LeRobot v2.1 Episode Dataset Generator (`lerobotDatasetGenerator.ts`)
 *
 * Turns any calibrated LetThereBeSims tabletop scene + verified controller into
 * an imitation-learning dataset formatted to the Hugging Face LeRobot v2.1
 * layout so roboticists can immediately train ACT, Diffusion Policy, or SmolVLA:
 *
 * 1. Executes N domain-randomized pick-lift-transport-place episodes through the
 *    calibrated rigid-body physics & 6-DOF IK solver (varying tabletop friction,
 *    target mass, and XY pose offsets across a seeded spread).
 * 2. Automatically filters out failed rollouts (unless `includeFailedEpisodes`
 *    is explicitly toggled), tracking attempted vs. passed vs. filtered counts.
 * 3. Resamples every episode to the requested control frequency (`10/20/30 Hz`),
 *    computes authentic per-feature normalization stats (`min`, `max`, `mean`,
 *    `std` for `observation.state`, `action`, `observation.environment_state`,
 *    and `next.reward`), and emits:
 *    - `meta/info.json`
 *    - `meta/tasks.jsonl`
 *    - `meta/episodes.jsonl`
 *    - `meta/episodes_stats.jsonl`
 *    - `data/chunk-000/episode_000000.csv` ... `episode_XXXXXX.csv`
 *    - `scene.mjcf.xml` (for offline MuJoCo multi-camera `.mp4` rendering)
 *    - `scripts/build_parquet_and_videos.py` (1-command Parquet + MP4 builder)
 *    - `scripts/train_lerobot.sh` (ACT / Diffusion Policy / SmolVLA CLI launcher)
 *    - `README.md` (Hugging Face Dataset Card)
 *
 * Use cases:
 * - Scaling 1 real-world phone video into 25–100 verified synthetic training
 *   demonstrations ready for LeRobot fine-tuning.
 */

import { APP_CONFIG, type RobotModelId } from '../config/appConfig';
import type {
  ControllerParams,
  DomainRandomizationParam,
  LeRobotDatasetPreviewResponse,
  LeRobotEpisodeSummary,
  LeRobotFeatureStat,
  LeRobotPolicyTarget,
  LeRobotRandomizationSpread,
  LeRobotSampleFrame,
  RolloutResult,
  SceneSpec,
  TrainingPackageFile,
  TrajectoryFrame,
} from '../types/robotics';
import { logFunctionCall } from '../utils/logger';
import { resolveTaskGoalPose } from '../utils/taskGoalResolver';
import { buildZipArchive, type ZipEntryInput } from '../utils/zipBuilder';
import { buildRlReadyMjcfXml } from './gymMjxExporter';
import { simulateSingleEnvironment } from './physicsEvaluator';

export interface LeRobotGeneratorInput {
  sourceLabel: string;
  taskGoal: string;
  robotModel: RobotModelId;
  sceneSpec: SceneSpec;
  controllerParams: ControllerParams;
  episodeCount: number;
  fps: number;
  policyTarget: LeRobotPolicyTarget;
  randomizationSpread: LeRobotRandomizationSpread;
  includeFailedEpisodes: boolean;
}

interface RecordedFrame {
  frameIndex: number;
  episodeIndex: number;
  globalIndex: number;
  timestamp: number;
  state: number[]; // 7-D: 6 joints + gripper aperture
  action: number[]; // 7-D: next-step target joints + gripper
  envState: number[]; // 9-D: ee(3) + obj(3) + goal(2) + force(1)
  contactForceN: number;
  reward: number;
  done: boolean;
}

interface GeneratedEpisode {
  summary: LeRobotEpisodeSummary;
  frames: RecordedFrame[];
  stats: Record<string, { min: number[]; max: number[]; mean: number[]; std: number[] }>;
}

/**
 * Deterministic mulberry32 PRNG so dataset generation is reproducible per seed.
 */
function createSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Converts a name to a clean lowercase slug for dataset and repo naming.
 */
function toSlug(raw: string): string {
  const cleaned = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 32);
  return cleaned || 'tabletop_pick_place';
}

/**
 * Linearly interpolates between two `TrajectoryFrame` samples at factor `t` in [0, 1].
 */
function lerpFrame(a: TrajectoryFrame, b: TrajectoryFrame, t: number): TrajectoryFrame {
  const mix = (u: number, v: number) => u + (v - u) * t;
  return {
    timeSec: mix(a.timeSec, b.timeSec),
    jointAngles: [
      mix(a.jointAngles[0], b.jointAngles[0]),
      mix(a.jointAngles[1], b.jointAngles[1]),
      mix(a.jointAngles[2], b.jointAngles[2]),
      mix(a.jointAngles[3], b.jointAngles[3]),
      mix(a.jointAngles[4], b.jointAngles[4]),
      mix(a.jointAngles[5], b.jointAngles[5]),
    ],
    gripperOpen: mix(a.gripperOpen, b.gripperOpen),
    endEffectorPos: {
      x: mix(a.endEffectorPos.x, b.endEffectorPos.x),
      y: mix(a.endEffectorPos.y, b.endEffectorPos.y),
      z: mix(a.endEffectorPos.z, b.endEffectorPos.z),
    },
    objectPos: {
      x: mix(a.objectPos.x, b.objectPos.x),
      y: mix(a.objectPos.y, b.objectPos.y),
      z: mix(a.objectPos.z, b.objectPos.z),
    },
    objectTiltRad: mix(a.objectTiltRad, b.objectTiltRad),
    contactForceN: mix(a.contactForceN, b.contactForceN),
    slipVelocityMs: mix(a.slipVelocityMs, b.slipVelocityMs),
  };
}

/**
 * Resamples a simulated rollout trajectory to an exact uniform frame rate (`fps`).
 */
function resampleRolloutAtFps(
  rollout: RolloutResult,
  fps: number,
  goalX: number,
  goalZ: number,
  episodeIndex: number,
  startGlobalIndex: number,
): RecordedFrame[] {
  const src = rollout.trajectory;
  if (src.length < 2) return [];

  const durationSec = Math.max(1.0, rollout.durationSec ?? src[src.length - 1].timeSec);
  const frameCount = Math.max(12, Math.round(durationSec * fps));
  const resampled: TrajectoryFrame[] = [];

  for (let i = 0; i < frameCount; i++) {
    const frac = i / Math.max(1, frameCount - 1);
    const exactIdx = frac * (src.length - 1);
    const lo = Math.floor(exactIdx);
    const hi = Math.min(src.length - 1, lo + 1);
    resampled.push(lerpFrame(src[lo], src[hi], exactIdx - lo));
  }

  const frames: RecordedFrame[] = [];
  for (let i = 0; i < resampled.length; i++) {
    const cur = resampled[i];
    const nxt = resampled[Math.min(resampled.length - 1, i + 1)];
    const state = [
      ...cur.jointAngles.map((v) => Number(v.toFixed(5))),
      Number(cur.gripperOpen.toFixed(4)),
    ];
    const action = [
      ...nxt.jointAngles.map((v) => Number(v.toFixed(5))),
      Number(nxt.gripperOpen.toFixed(4)),
    ];
    const envState = [
      Number(cur.endEffectorPos.x.toFixed(4)),
      Number(cur.endEffectorPos.y.toFixed(4)),
      Number(cur.endEffectorPos.z.toFixed(4)),
      Number(cur.objectPos.x.toFixed(4)),
      Number(cur.objectPos.y.toFixed(4)),
      Number(cur.objectPos.z.toFixed(4)),
      Number(goalX.toFixed(4)),
      Number(goalZ.toFixed(4)),
      Number(cur.contactForceN.toFixed(3)),
    ];

    const distToGoal = Math.hypot(cur.objectPos.x - goalX, cur.objectPos.z - goalZ);
    const isLast = i === resampled.length - 1;
    const reward = Number(
      (
        Math.exp(-4.0 * distToGoal) * 0.5 +
        (isLast && rollout.success ? 10.0 : 0.0) -
        Math.max(0, cur.contactForceN - APP_CONFIG.simulation.maxSafeContactForceN) * 0.05
      ).toFixed(4),
    );

    frames.push({
      frameIndex: i,
      episodeIndex,
      globalIndex: startGlobalIndex + i,
      timestamp: Number((i / fps).toFixed(4)),
      state,
      action,
      envState,
      contactForceN: Number(cur.contactForceN.toFixed(3)),
      reward,
      done: isLast,
    });
  }

  return frames;
}

/**
 * Computes element-wise min, max, mean, and std across a matrix of equal-length vectors.
 */
function computeVectorStats(vectors: number[][]): {
  min: number[];
  max: number[];
  mean: number[];
  std: number[];
} {
  if (vectors.length === 0) {
    return { min: [], max: [], mean: [], std: [] };
  }
  const dims = vectors[0].length;
  const min = new Array<number>(dims).fill(Infinity);
  const max = new Array<number>(dims).fill(-Infinity);
  const sum = new Array<number>(dims).fill(0);

  for (const vec of vectors) {
    for (let d = 0; d < dims; d++) {
      const v = vec[d];
      if (v < min[d]) min[d] = v;
      if (v > max[d]) max[d] = v;
      sum[d] += v;
    }
  }

  const mean = sum.map((s) => s / vectors.length);
  const sqDiff = new Array<number>(dims).fill(0);
  for (const vec of vectors) {
    for (let d = 0; d < dims; d++) {
      const diff = vec[d] - mean[d];
      sqDiff[d] += diff * diff;
    }
  }
  const std = sqDiff.map((s) => Math.sqrt(s / Math.max(1, vectors.length)));

  const roundArr = (arr: number[]) => arr.map((v) => Number(v.toFixed(5)));
  return {
    min: roundArr(min),
    max: roundArr(max),
    mean: roundArr(mean),
    std: roundArr(std),
  };
}

/**
 * Generates an episode CSV string matching the flat LeRobot v2.1 tabular columns.
 */
function serializeEpisodeCsv(frames: RecordedFrame[]): string {
  const header = [
    'episode_index',
    'frame_index',
    'index',
    'task_index',
    'timestamp',
    'obs_state_j0',
    'obs_state_j1',
    'obs_state_j2',
    'obs_state_j3',
    'obs_state_j4',
    'obs_state_j5',
    'obs_state_gripper',
    'action_j0',
    'action_j1',
    'action_j2',
    'action_j3',
    'action_j4',
    'action_j5',
    'action_gripper',
    'env_ee_x',
    'env_ee_y',
    'env_ee_z',
    'env_obj_x',
    'env_obj_y',
    'env_obj_z',
    'env_goal_x',
    'env_goal_z',
    'env_contact_force_n',
    'next_reward',
    'next_done',
  ].join(',');

  const rows = frames.map((f) =>
    [
      f.episodeIndex,
      f.frameIndex,
      f.globalIndex,
      0,
      f.timestamp,
      ...f.state,
      ...f.action,
      ...f.envState,
      f.reward,
      f.done ? 1 : 0,
    ].join(','),
  );
  return `${header}\n${rows.join('\n')}\n`;
}

/**
 * Generates `scripts/build_parquet_and_videos.py` to convert exported CSV episodes
 * into native LeRobot `.parquet` files and optional MuJoCo multi-camera `.mp4` clips.
 */
function buildParquetAndVideoScript(params: {
  datasetName: string;
  fps: number;
  robotModel: RobotModelId;
}): string {
  const { datasetName, fps, robotModel } = params;
  return `"""
LeRobot v2.1 Parquet + Multi-Camera MuJoCo Video Builder
Dataset: ${datasetName} (${APP_CONFIG.robots[robotModel].name} @ ${fps} FPS)

Converts every exported \`data/chunk-000/episode_*.csv\` into native LeRobot
Parquet files with list-typed \`observation.state\` / \`action\` columns, and
optionally renders synchronized \`wrist_cam\` + \`overhead_cam\` MP4 videos from
\`scene.mjcf.xml\` using the exact recorded joint trajectories.

Usage:
  pip install pandas pyarrow mujoco imageio[ffmpeg]
  python scripts/build_parquet_and_videos.py --render-videos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "chunk-000"
FPS = ${fps}


def convert_episode_csv_to_parquet(csv_path: Path) -> pd.DataFrame:
    """Converts a flat LetThereBeSims CSV episode into native LeRobot v2.1 Parquet."""
    df = pd.read_csv(csv_path)
    state_cols = [f"obs_state_j{i}" for i in range(6)] + ["obs_state_gripper"]
    action_cols = [f"action_j{i}" for i in range(6)] + ["action_gripper"]
    env_cols = [
        "env_ee_x", "env_ee_y", "env_ee_z",
        "env_obj_x", "env_obj_y", "env_obj_z",
        "env_goal_x", "env_goal_z", "env_contact_force_n",
    ]

    table = pa.table({
        "observation.state": pa.array(df[state_cols].astype(np.float32).values.tolist(), type=pa.list_(pa.float32(), 7)),
        "action": pa.array(df[action_cols].astype(np.float32).values.tolist(), type=pa.list_(pa.float32(), 7)),
        "observation.environment_state": pa.array(df[env_cols].astype(np.float32).values.tolist(), type=pa.list_(pa.float32(), 9)),
        "timestamp": pa.array(df["timestamp"].astype(np.float32)),
        "frame_index": pa.array(df["frame_index"].astype(np.int64)),
        "episode_index": pa.array(df["episode_index"].astype(np.int64)),
        "index": pa.array(df["index"].astype(np.int64)),
        "task_index": pa.array(df["task_index"].astype(np.int64)),
        "next.reward": pa.array(df["next_reward"].astype(np.float32)),
        "next.done": pa.array(df["next_done"].astype(bool)),
    })

    out_path = csv_path.with_suffix(".parquet")
    pq.write_table(table, out_path)
    return df


def render_mujoco_cameras(df: pd.DataFrame, episode_idx: int) -> None:
    """Renders synchronized wrist_cam & overhead_cam MP4 clips from scene.mjcf.xml."""
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.mjcf.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    target_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_freejoint")
    q_adr = int(model.jnt_qposadr[target_jnt])

    for cam_name, folder_key in [("wrist_cam", "observation.images.wrist"), ("overhead_cam", "observation.images.top")]:
        vid_dir = ROOT / "videos" / "chunk-000" / folder_key
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"episode_{episode_idx:06d}.mp4"
        frames = []
        for _, row in df.iterrows():
            data.qpos[:6] = [row[f"obs_state_j{i}"] for i in range(6)]
            grip_m = -0.018 + float(row["obs_state_gripper"]) * 0.040
            data.qpos[6] = grip_m
            data.qpos[7] = grip_m
            data.qpos[q_adr : q_adr + 3] = [row["env_obj_x"], -row["env_obj_z"], max(0.02, row["env_obj_y"])]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam_name)
            frames.append(renderer.render())
        imageio.mimsave(vid_path, frames, fps=FPS)
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-videos", action="store_true", help="Render MuJoCo wrist + overhead MP4 videos")
    args = parser.parse_args()

    csv_files = sorted(DATA_DIR.glob("episode_*.csv"))
    print(f"Converting {len(csv_files)} episodes to LeRobot v2.1 Parquet...")
    for idx, csv_file in enumerate(csv_files):
        df = convert_episode_csv_to_parquet(csv_file)
        if args.render_videos:
            render_mujoco_cameras(df, idx)
    print("Done! Dataset is ready for \`lerobot-train\`.")


if __name__ == "__main__":
    main()
`;
}

/**
 * Generates `scripts/train_lerobot.sh` customized for ACT, Diffusion Policy, or SmolVLA.
 */
function buildTrainLeRobotShell(params: {
  repoIdSuggestion: string;
  policyTarget: LeRobotPolicyTarget;
  datasetName: string;
}): string {
  const { repoIdSuggestion, policyTarget, datasetName } = params;
  const policyType =
    policyTarget === 'diffusion'
      ? 'diffusion'
      : policyTarget === 'smolvla'
        ? 'smolvla'
        : 'act';
  const extraFlags =
    policyTarget === 'smolvla'
      ? ' \\\n  --policy.path=lerobot/smolvla_base \\\n  --steps=20000 \\\n  --batch_size=16'
      : policyTarget === 'diffusion'
        ? ' \\\n  --steps=50000 \\\n  --batch_size=32'
        : ' \\\n  --steps=40000 \\\n  --batch_size=16';

  return `#!/usr/bin/env bash
set -euo pipefail

# LetThereBeSims -> Hugging Face LeRobot v2.1 Training Launcher
# Policy Architecture: ${policyType.toUpperCase()}
# Dataset: ${datasetName}

# 1. Build native Parquet files (+ optional MuJoCo wrist/overhead MP4 cameras)
python3 scripts/build_parquet_and_videos.py --render-videos

# 2. Launch LeRobot policy training (install via: pip install lerobot)
lerobot-train \\
  --dataset.repo_id="${repoIdSuggestion}" \\
  --dataset.root="$(pwd)" \\
  --policy.type=${policyType} \\
  --output_dir="outputs/train/${datasetName}_${policyType}" \\
  --job_name="${datasetName}_${policyType}" \\
  --policy.device=cuda${extraFlags}
`;
}

/**
 * Executes the core synthetic episode batch and builds all dataset files + metrics.
 */
function generateLeRobotBundleCore(input: LeRobotGeneratorInput): {
  preview: LeRobotDatasetPreviewResponse;
  zipEntries: ZipEntryInput[];
  zipFilename: string;
} {
  const targetCount = Math.max(
    1,
    Math.min(APP_CONFIG.trainingExport.lerobot.maxEpisodeCount, Math.round(input.episodeCount)),
  );
  const fps = APP_CONFIG.trainingExport.lerobot.fpsPresets.includes(input.fps)
    ? input.fps
    : APP_CONFIG.trainingExport.lerobot.defaultFps;
  const episodeDurationSec = APP_CONFIG.trainingExport.lerobot.defaultEpisodeDurationSec;

  const spreadMult =
    input.randomizationSpread === 'low'
      ? 0.55
      : input.randomizationSpread === 'high'
        ? 1.45
        : 1.0;

  const goalPad = resolveTaskGoalPose(input.sceneSpec);
  const baseObj = input.sceneSpec.targetObject;
  const rng = createSeededRng(
    targetCount * 997 + fps * 31 + Math.round(baseObj.position.x * 1000) + 1337,
  );

  // Oversample slightly when filtering failed episodes so we hit the requested count cleanly
  const maxAttempts = input.includeFailedEpisodes
    ? targetCount
    : Math.min(targetCount * 2 + 10, 260);

  const episodes: GeneratedEpisode[] = [];
  let attempted = 0;
  let filteredOut = 0;
  let globalFrameCursor = 0;

  for (let attempt = 0; attempt < maxAttempts && episodes.length < targetCount; attempt++) {
    attempted++;
    const friction = Number(
      Math.max(0.18, Math.min(1.25, baseObj.friction + (rng() * 2 - 1) * 0.16 * spreadMult)).toFixed(3),
    );
    const mass = Number(
      Math.max(0.015, Math.min(2.2, baseObj.massKg * (1 + (rng() * 2 - 1) * 0.28 * spreadMult))).toFixed(3),
    );
    const dx = Number(((rng() * 2 - 1) * 0.032 * spreadMult).toFixed(4));
    const dz = Number(((rng() * 2 - 1) * 0.032 * spreadMult).toFixed(4));

    const randParam: DomainRandomizationParam = {
      id: attempt + 1,
      friction,
      mass,
      xyOffset: [dx, dz],
    };

    const rollout = simulateSingleEnvironment(
      input.sceneSpec,
      randParam,
      input.controllerParams,
      input.robotModel,
      episodeDurationSec,
    );

    if (!rollout.success && !input.includeFailedEpisodes) {
      filteredOut++;
      continue;
    }

    const epIdx = episodes.length;
    const frames = resampleRolloutAtFps(
      rollout,
      fps,
      goalPad.x,
      goalPad.z,
      epIdx,
      globalFrameCursor,
    );
    if (frames.length === 0) continue;

    globalFrameCursor += frames.length;
    const totalReward = Number(frames.reduce((acc, f) => acc + f.reward, 0).toFixed(2));
    const stateStats = computeVectorStats(frames.map((f) => f.state));
    const actionStats = computeVectorStats(frames.map((f) => f.action));
    const envStats = computeVectorStats(frames.map((f) => f.envState));
    const rewardStats = computeVectorStats(frames.map((f) => [f.reward]));

    episodes.push({
      summary: {
        episodeIndex: epIdx,
        lengthFrames: frames.length,
        durationSec: Number((frames.length / fps).toFixed(2)),
        passed: rollout.success,
        friction,
        massKg: mass,
        xyOffset: [dx, dz],
        peakContactForceN: Number(rollout.peakContactForceN.toFixed(2)),
        maxSlipVelocityMs: Number(rollout.maxSlipVelocityMs.toFixed(4)),
        placementErrorM: Number((rollout.placementErrorM ?? 0.004).toFixed(4)),
        totalReward,
      },
      frames,
      stats: {
        'observation.state': stateStats,
        action: actionStats,
        'observation.environment_state': envStats,
        'next.reward': rewardStats,
      },
    });
  }

  // Compute global feature normalization stats across all exported frames
  const allFrames = episodes.flatMap((e) => e.frames);
  const globalStateStats = computeVectorStats(allFrames.map((f) => f.state));
  const globalActionStats = computeVectorStats(allFrames.map((f) => f.action));
  const globalEnvStats = computeVectorStats(allFrames.map((f) => f.envState));
  const globalRewardStats = computeVectorStats(allFrames.map((f) => [f.reward]));

  const featureStats: LeRobotFeatureStat[] = [
    { featureName: 'observation.state', shape: [7], ...globalStateStats },
    { featureName: 'action', shape: [7], ...globalActionStats },
    { featureName: 'observation.environment_state', shape: [9], ...globalEnvStats },
    { featureName: 'next.reward', shape: [1], ...globalRewardStats },
  ];

  const slug = toSlug(baseObj.name);
  const datasetName = `lettherebesims_${slug}_lerobot`;
  const repoIdSuggestion = `local/${datasetName}`;

  const infoJsonObj = {
    codebase_version: APP_CONFIG.trainingExport.lerobot.codebaseVersion,
    robot_type: input.robotModel,
    total_episodes: episodes.length,
    total_frames: allFrames.length,
    total_tasks: 1,
    total_videos: 0,
    total_chunks: 1,
    chunks_size: 1000,
    fps,
    splits: {
      train: `0:${episodes.length}`,
    },
    data_path: 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
    video_path: 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
    features: {
      'observation.state': {
        dtype: 'float32',
        shape: [7],
        names: [
          'base_yaw',
          'shoulder_pitch',
          'elbow_pitch',
          'wrist_pitch',
          'wrist_roll',
          'flange_yaw',
          'gripper_open',
        ],
      },
      action: {
        dtype: 'float32',
        shape: [7],
        names: [
          'target_base_yaw',
          'target_shoulder_pitch',
          'target_elbow_pitch',
          'target_wrist_pitch',
          'target_wrist_roll',
          'target_flange_yaw',
          'target_gripper_open',
        ],
      },
      'observation.environment_state': {
        dtype: 'float32',
        shape: [9],
        names: [
          'ee_x',
          'ee_y',
          'ee_z',
          'obj_x',
          'obj_y',
          'obj_z',
          'goal_x',
          'goal_z',
          'contact_force_n',
        ],
      },
      timestamp: { dtype: 'float32', shape: [1], names: null },
      frame_index: { dtype: 'int64', shape: [1], names: null },
      episode_index: { dtype: 'int64', shape: [1], names: null },
      index: { dtype: 'int64', shape: [1], names: null },
      task_index: { dtype: 'int64', shape: [1], names: null },
      'next.reward': { dtype: 'float32', shape: [1], names: null },
      'next.done': { dtype: 'bool', shape: [1], names: null },
    },
  };

  const infoJson = `${JSON.stringify(infoJsonObj, null, 2)}\n`;
  const tasksJsonl = `${JSON.stringify({ task_index: 0, task: input.taskGoal })}\n`;
  const episodesJsonl = `${episodes
    .map((e) =>
      JSON.stringify({
        episode_index: e.summary.episodeIndex,
        tasks: [input.taskGoal],
        length: e.summary.lengthFrames,
        passed: e.summary.passed,
        domain_randomization: {
          friction: e.summary.friction,
          mass_kg: e.summary.massKg,
          xy_offset: e.summary.xyOffset,
        },
        peak_contact_force_n: e.summary.peakContactForceN,
        placement_error_m: e.summary.placementErrorM,
      }),
    )
    .join('\n')}\n`;

  const episodesStatsJsonl = `${episodes
    .map((e) =>
      JSON.stringify({
        episode_index: e.summary.episodeIndex,
        stats: e.stats,
      }),
    )
    .join('\n')}\n`;

  const buildScriptPy = buildParquetAndVideoScript({
    datasetName,
    fps,
    robotModel: input.robotModel,
  });
  const trainShell = buildTrainLeRobotShell({
    repoIdSuggestion,
    policyTarget: input.policyTarget,
    datasetName,
  });
  const mjcfXml = buildRlReadyMjcfXml({
    sceneSpec: input.sceneSpec,
    robotModel: input.robotModel,
    taskGoal: input.taskGoal,
  });

  const readmeMd = `---
task_categories:
  - robotics
tags:
  - LeRobot
  - LetThereBeSims
  - MuJoCo
  - ${input.robotModel}
---

# ${datasetName} (LeRobot ${APP_CONFIG.trainingExport.lerobot.codebaseVersion} Synthetic Dataset)

- **Robot:** ${APP_CONFIG.robots[input.robotModel].name}
- **Task Goal:** ${input.taskGoal}
- **Episodes:** ${episodes.length} verified episodes (${allFrames.length.toLocaleString()} frames @ ${fps} Hz)
- **Target Policy:** \`${input.policyTarget.toUpperCase()}\`

## Build Native Parquet + Multi-Camera Videos & Train

\`\`\`bash
pip install pandas pyarrow mujoco "imageio[ffmpeg]" lerobot

# Convert CSV episodes to native LeRobot v2.1 Parquet + render MuJoCo wrist/overhead cameras
python3 scripts/build_parquet_and_videos.py --render-videos

# Launch policy training (${input.policyTarget.toUpperCase()})
bash scripts/train_lerobot.sh
\`\`\`
`;

  const zipEntries: ZipEntryInput[] = [
    { path: `${datasetName}/meta/info.json`, content: infoJson },
    { path: `${datasetName}/meta/tasks.jsonl`, content: tasksJsonl },
    { path: `${datasetName}/meta/episodes.jsonl`, content: episodesJsonl },
    { path: `${datasetName}/meta/episodes_stats.jsonl`, content: episodesStatsJsonl },
    { path: `${datasetName}/scripts/build_parquet_and_videos.py`, content: buildScriptPy },
    { path: `${datasetName}/scripts/train_lerobot.sh`, content: trainShell },
    { path: `${datasetName}/scene.mjcf.xml`, content: mjcfXml },
    { path: `${datasetName}/README.md`, content: readmeMd },
  ];

  for (const ep of episodes) {
    const padded = String(ep.summary.episodeIndex).padStart(6, '0');
    zipEntries.push({
      path: `${datasetName}/data/chunk-000/episode_${padded}.csv`,
      content: serializeEpisodeCsv(ep.frames),
    });
  }

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

  const previewFiles: TrainingPackageFile[] = [
    makeFile('meta/info.json', 'meta/info.json', 'json', infoJson),
    makeFile('scripts/train_lerobot.sh', 'scripts/train_lerobot.sh', 'bash', trainShell),
    makeFile(
      'scripts/build_parquet_and_videos.py',
      'Parquet & Video Builder (.py)',
      'python',
      buildScriptPy,
    ),
    makeFile('meta/episodes.jsonl', 'meta/episodes.jsonl', 'json', episodesJsonl),
    makeFile('README.md', 'Dataset Card (README.md)', 'markdown', readmeMd),
  ];

  const passedEpisodesCount = episodes.filter((e) => e.summary.passed).length;
  const meanPeakForceN =
    episodes.length > 0
      ? Number(
          (
            episodes.reduce((acc, e) => acc + e.summary.peakContactForceN, 0) / episodes.length
          ).toFixed(2),
        )
      : 0;
  const meanPlacementErrorMm =
    episodes.length > 0
      ? Number(
          (
            (episodes.reduce((acc, e) => acc + e.summary.placementErrorM, 0) / episodes.length) *
            1000
          ).toFixed(1),
        )
      : 0;

  const sampleFrames: LeRobotSampleFrame[] = (episodes[0]?.frames ?? [])
    .slice(0, 12)
    .map((f) => ({
      frameIndex: f.frameIndex,
      timestamp: f.timestamp,
      state: f.state,
      action: f.action,
      contactForceN: f.contactForceN,
      reward: f.reward,
      done: f.done,
    }));

  const preview: LeRobotDatasetPreviewResponse = {
    datasetName,
    repoIdSuggestion,
    sourceLabel: input.sourceLabel,
    robotModel: input.robotModel,
    taskGoal: input.taskGoal,
    codebaseVersion: APP_CONFIG.trainingExport.lerobot.codebaseVersion,
    fps,
    policyTarget: input.policyTarget,
    randomizationSpread: input.randomizationSpread,
    attemptedEpisodes: attempted,
    exportedEpisodes: episodes.length,
    filteredOutEpisodes: filteredOut,
    totalFrames: allFrames.length,
    successRatePct: Number(((passedEpisodesCount / Math.max(1, attempted)) * 100).toFixed(1)),
    meanPeakForceN,
    meanPlacementErrorMm,
    episodes: episodes.map((e) => e.summary),
    featureStats,
    sampleFrames,
    previewFiles,
    generatedAt: new Date().toISOString(),
  };

  return {
    preview,
    zipEntries,
    zipFilename: `${datasetName}.zip`,
  };
}

/**
 * Generates a synthetic LeRobot v2.1 dataset preview, episode quality report,
 * normalization stats, and key metadata/script previews.
 */
export function buildLeRobotDatasetPreview(
  input: LeRobotGeneratorInput,
): LeRobotDatasetPreviewResponse {
  logFunctionCall('buildLeRobotDatasetPreview', {
    sourceLabel: input.sourceLabel,
    robotModel: input.robotModel,
    episodeCount: input.episodeCount,
    fps: input.fps,
    policyTarget: input.policyTarget,
    randomizationSpread: input.randomizationSpread,
    includeFailedEpisodes: input.includeFailedEpisodes,
  });
  return generateLeRobotBundleCore(input).preview;
}

/**
 * Compiles the full synthetic LeRobot v2.1 episode dataset into a `.zip` archive.
 */
export function buildLeRobotDatasetZip(input: LeRobotGeneratorInput): {
  filename: string;
  buffer: Buffer;
} {
  logFunctionCall('buildLeRobotDatasetZip', {
    sourceLabel: input.sourceLabel,
    episodeCount: input.episodeCount,
    fps: input.fps,
    policyTarget: input.policyTarget,
  });
  const { zipEntries, zipFilename } = generateLeRobotBundleCore(input);
  return {
    filename: zipFilename,
    buffer: buildZipArchive(zipEntries),
  };
}
