# LetThereBeSims — Application Design Document

## 1. Purpose & Overview

**LetThereBeSims** is a Sim-to-Real Robotics "Failure-to-Policy" Compiler designed to close the gap between real-world tabletop manipulation footage and verified robotic control policies. A robotics engineer or live demo evaluator scans a QR code from the desktop command center using a smartphone (or uploads/records 1 or 2 camera views directly on desktop), submits a clip of a cluttered tabletop, successful run, or failed grasp, and watches a stateful **Gemini 4 Interactions API** pipeline reconstruct the physical workspace, simulate 20 domain-randomized environments, diagnose contact anomalies, patch the Python control policy in a closed loop until pass rates reach **20/20 (100%)**, and compile a complete deployment bundle.

---

## 2. Core Features

### 2.1 QR-Paired Mobile Capture (`/capture?session=<id>`)
- Every desktop workspace session generates a unique session identifier and renders a scannable QR code pointing to `/capture?session=<id>`.
- Scanning the QR code on iOS Safari or Android Chrome opens a mobile-first dark viewfinder interface with touch targets $\ge 48\text{px}$.
- Supports both native rear-camera capture (`accept="video/*" capture="environment"`) and live browser `MediaRecorder` recording.
- Uploads video over multipart HTTP (`POST /api/sessions/:id/upload`) and immediately notifies the paired desktop command center via Server-Sent Events (SSE).

### 2.2 Desktop Video Upload (1 or 2 Views) & Live Webcam Intake
- Multi-file drag-and-drop and picker (`Upload 1 or 2 Videos`) accepting `.mp4`, `.webm`, `.mov`, or frame images, plus a one-click live webcam recorder directly inside the left intake rail.
- Uploading **1 video** stages View 1 and keeps the 3D workspace in **Single View**; uploading **2 videos at the same time** (or clicking **Add Optional View 2**) stages View 1 + View 2 side-by-side, unlocks the **Dual 3D View** automatically, and feeds both camera angles into Gemini 4 Carbon for multi-view spatial fusion. Removing View 2 (`DELETE /api/sessions/:id/media?slot=secondary`) cleanly collapses back to Single View.
- Proxy-safe chunked binary upload over `window.fetch` (`1.5 MB` `application/octet-stream` chunks via `/upload/init`, `/upload/chunk`, and `/upload/complete`) with slot routing (`primary` / `secondary`) and automatic session restoration across restarts.
- Immediate local blob preview (`URL.createObjectURL`) paired with RFC 9110 byte-range server streaming (`/api/sessions/:id/media?slot=primary|secondary`) and automatic MIME-from-extension inference plus codec fallback notices.
- Task goal prompt input and robot kinematic selector (`SO-ARM101 6-DOF` and `Franka Emika Panda 7-DOF`).

### 2.3 Stateful Multi-Model Gemini 4 Pipeline (`previous_interaction_id`)
1. **Stage 1 — Spatial Real-to-Sim Architect (`gemini-4-carbon`, `thinking_level: "medium"`):**
   - Automatically transcodes large uploaded videos (`> 2.5 MB`) into a lightweight full-timeline `1 FPS / 480p` H.264 analysis proxy via `ffmpeg` (`src/server/videoTranscoder.ts`) while preserving the full-resolution original for browser playback, and routes payloads `> 2 MB` through the Gemini Files API (`> 2 MB` never sent as inline base64).
   - Runs via resilient `background: true` polling (`ai.interactions.get` every 3s with a 3-minute hard stage timeout) so multi-view or 1–2 minute clips never hang indefinitely on a dropped synchronous HTTP socket.
   - Ingests 1 or 2 uploaded camera views and explicitly classifies both robot presence (`hasRobotInVideo`) and **execution outcome (`executionOutcome: 'failure' | 'success' | 'no_robot'`)** with a human-readable `outcomeSummary`:
     - **`failure` (`robot_failure_repair`):** Diagnoses physical failure modes and launches the closed-loop 20-environment repair swarm.
     - **`success` (`robot_success_verification`):** Marks the clip as a verified successful demonstration, switches UI diagnostics to **Demonstration Robustness Analysis**, and stress-tests the nominal policy across 20 domain-randomized perturbations (short-circuiting Stage 3 automatically if all 20/20 pass on Iteration 1, or hardening edge-case variants if any fail).
     - **`no_robot` (`scene_map_only`):** Cleanly finishes after Stage 1, hides the 3D robot arm, displays the simulation-ready 3D scene alongside an interactive **2D Top-Down Metric Spatial Map**, and lets the user export MuJoCo XML or click **Mount Robot & Simulate** to run Stages 2–4 on demand.
   - Calibrates tabletop scale, 3D poses, surface friction, obstacle geometry (up to 7 mapped obstacles), spatial clearance/hazard zones, and a recommended robot mounting coordinate.
   - Emits valid simulation-ready MuJoCo `MJCF` XML, 20 domain-randomized perturbation specs, baseline Python controller (`execute_policy`), and structured kinematic control parameters.
2. **Stage 2 — 20-Environment Swarm & Telemetry Triage (`gemini-4-hydrogen`, `thinking_level: "low"`):**
   - Fans out all 20 randomized environments via either the configured **External Cloud MuJoCo Webhook** (Modal / Cloud Run) or the **Built-In Server Physics Evaluator**.
   - Chains from Carbon's `interaction.id` to cluster contact-force spikes (`> 45 N`), slip velocities (`> 0.08 m/s`), and insufficient Z-lift (`< 0.12 m`) or confirm robustness safety margins.
3. **Stage 3 — Adversarial Policy Patching Loop (`gemini-4-neon`, `thinking_level: "xhigh"`):**
   - Chains from Hydrogen's triage turn via resilient `background: true` polling with a 3-minute stage timeout to diagnose physical root causes and patch `execute_policy(model, data, step)` plus structured gain/approach parameters (or completes immediately at Iteration 1 if a successful demonstration already passes 20/20 variants).
   - Loops automatically with the 20-environment swarm until `20/20 (100%)` pass or the configured iteration cap is reached.
4. **Stage 4 — Asynchronous Long Decode Deployment Bundle (`gemini-4-neon`, `background: true`):**
   - Generates a standalone LeRobot/MuJoCo controller script with safety torque limits, a 20-scene MJCF regression suite, and a Sim-to-Real Contact Dynamics Audit Report.
5. **Real-Time Numeric Progress Bar, Elapsed Telemetry & Live Cancel (`pipelineProgress.ts`):**
   - Streams authoritative server-side `PipelineProgressState` milestones over SSE (`percent`, `targetPercent`, `expectedStageDurationSec`, `stageLabel`, `startedAt`) combined with bounded client-side asymptotic interpolation so the **Analyzing & Verifying Policy** button and **Gemini 4 Pipeline Rail** display a live monotonic `0–100%` progress bar, active stage description, elapsed `MM:SS` timer, typical run duration window, and an inline **Cancel** button (`POST /api/sessions/:id/cancel`) that aborts any in-flight run cleanly.

### 2.4 Interactive Dynamic 3D Robot Simulation Stage (Three.js / WebGL)
- Composite multi-part procedural **SO-ARM101** (and **Franka Panda**) articulated arms with wrist-mounted optical camera modules mounted on a dynamically scaled calibrated metric workbench with `RoomEnvironment` PBR lighting, warm/cool 3-point lighting, and software-WebGL pacing guards.
- **Complete Pick-Lift-Transport-Place Task Execution & Goal Placement Pad (`taskGoalResolver.ts`, `livePhysicsSandbox.ts`, `physicsEvaluator.ts`):**
  - Resolves a deterministic, obstacle-free **Goal Placement Pad** (`clear_zone` preference or open opposite tabletop zone within safe arm reach) shared identically by the backend 20-environment physics evaluator, the browser's real-time 6-DOF IK sandbox, and the 3D workbench Goal Pad reticle.
  - Executes the complete 8-phase manipulation task (`Approach → Descend → Grasp → Lift → Transport over obstacles → Controlled Place & Release on Goal Pad → Retract → Task Complete ✓`) in both Live Physics mode and Replay mode, glowing bright emerald with a completion shockwave pulse when the object is cleanly placed at the destination.
  - **Direct 3D Tabletop Drag-and-Drop (`stageInteractionController.ts`):** Raycast hover highlighting and ray-plane tabletop dragging let users grab and reposition the target object or any obstacle directly in 3D space (automatically handing off `OrbitControls` and transitioning into Live Physics mode) to test tighter obstacle corridors, collisions, and real-time IK re-planning.
  - **Live Impulse & Physics Tuning:** Built-in **Nudge** button imparts horizontal velocity and angular wobble onto the target object under Coulomb friction and gravity, while an inline **Physics** drawer inside the bottom HUD dock exposes live **Friction (μ)**, **Mass (kg)**, **Clearance (cm)** sliders, and **Reset Scene**.
  - **Dynamic Visual Telemetry:** Recomputes the full pickup-to-goal 3D spline in real time as objects move, dynamically scales the tabletop to match Carbon's calibrated `tableDimensions`, animates expanding shockwave rings on contact/collision/task-completion events, and supports an **Expand / Compact** height toggle (`480px` ↔ `660px`).
- **Full-Clip & 1-Minute Rollout Durations + 3D Video Clip Export (`stageVideoRecorder.ts`):**
  - Automatically detects the uploaded video's real duration (`Full Clip` mode, clamped cleanly to `3s–120s`) or lets the user select explicit **`15s`**, **`30s`**, or **`60s (1m)`** rollout lengths before compiling, executing continuous multi-cycle pick-lift-transport-place-return trajectories in true real-time playback with dynamic `TelemetryChart` time axes.
  - Built-in **Record** control in the bottom HUD dock captures the live Three.js WebGL canvas at 30 FPS (`video/webm`) for **`15s`**, **`30s`**, **`60s (1 min)`**, or the full rollout duration and downloads the video clip automatically.
- **Upload-Driven Viewport Mode (`stageViewportRenderer.ts`):**
  - **1 Video Uploaded (`Single` View — Default):** Renders a clean full-bleed single 3D viewport with no secondary split pane.
  - **2 Videos Uploaded (Optional Dual View Auto-Unlocked):** Automatically activates single-context `setScissorTest` / `setViewport` split rendering and reveals the `Dual Cam | Dual Arms | Single` toolbar (`Dual Cam` Workcell + live Wrist Eye-in-Hand Cam, `Dual Arms` SO-ARM101 vs. Franka Panda side-by-side, or `Single`).
- Automatically hides the robot arm and displays a calibrated amber mounting reticle in **Scene Map Mode** when the uploaded video contains an environment without a robot.
- Features a **collapsible bottom HUD dock** (Play/Pause, Live/Replay switcher, timeline scrubber or Nudge/Reset, Record 3D Clip bar, Physics tuning bar, playback speed, optional Dual Cam / Dual Arms / Single view switcher when 2 videos are staged, camera presets, vector/trajectory/grid toggles, Picture-in-Picture single or dual source footage toggle, Cloud MP4 toggle, and stage Expand/Compact toggle).

### 2.5 2D Spatial Scene Map, 20-Environment Matrix & Telemetry Chart
- **Interactive 2D Top-Down Metric Spatial Map:** Displays calibrated workspace bounds (cm), target & obstacle footprints with safety clearance rings, spatial zones (`robot_mount`, `clear_zone`, `hazard_zone`), reachable robot envelope, click-to-inspect object specs, 1-click MJCF export, and on-demand **Mount Robot & Simulate** trigger.
- Interactive 5×4 matrix (`E01`–`E20`) displaying each environment's pass/fail status with explicit text/glyph indicators, peak contact force (`N`), and final lift height (`m`), with `All` / `Failed` / `Passed` filtering and iteration switching.
- Synchronized SVG dual-axis telemetry chart plotting Contact Force (`N`, with `45.0 N` safety ceiling) and Object Z-Height (`m`, with `0.12 m` target threshold) with an interactive hover crosshair.

### 2.6 Failure-to-Regression-Suite CI/CD ("`pytest` for Physical Robots") & Contextual Views
- **Failure-to-Regression-Suite CI/CD (`regressionSuiteStore.ts`, `RegressionCiView.tsx`):**
  - Automatically promotes completed real-world failure repairs (or any compiled session via **Save to CI**) into a disk-persisted regression bank (`GET/POST/DELETE /api/regression/cases`).
  - Runs the full CI gate (`POST /api/regression/run`) across every saved failure case × 20 domain-randomized environments against the active session's policy, stored verified patches, or unpatched baselines—returning `ciExitCode: 0 (PASS)` or `1 (FAIL)` and logging chronological build history.
  - Generates ready-to-commit `.github/workflows/robotics-regression.yml`, `tests/test_robotics_regression.py`, and `tests/regression_suite_manifest.json` (`GET /api/regression/ci-bundle`) with one-click Copy & Download.
- **Command Center Export Bar:** Provides always-accessible **Save to CI** and **Export** buttons directly above the 3D stage to promote failures to CI or download the full session snapshot (`.json`), the 20-environment swarm telemetry table (`.csv`), the latest verified Python policy (`.py`), or the compiled MuJoCo scene (`.mjcf.xml`).
- **Policy Diff & MJCF XML Inspector:** Line-by-line diff comparing previous and patched Python controllers alongside Neon's root-cause explanation and Carbon's compiled MuJoCo XML.
- **Cloud Swarm Webhook Connector:** Allows connecting an external HTTPS Modal or Cloud Run MuJoCo swarm worker with server-side URL safety validation, connection tester, and ready-to-copy Modal endpoint schema.
- **Deployment Bundle Exporter:** Polls asynchronous Long Decode progress and provides one-click `.py`, `.xml`, and `.md` downloads.

### 2.7 Robot Training & Export Studio (`Train & Export` Tab)
- **Exportable Gymnasium + MuJoCo MJX RL Training Package (`gymMjxExporter.ts`, `zipBuilder.ts`, `TrainingExportView.tsx`):**
  - Generates a complete, runnable RL training bundle from either the active session or any saved Regression CI case (`POST /api/training/gym-mjx/preview` and `/download`).
  - Bundles 6 calibrated files into a single standard `.zip` archive: RL-ready `scene.mjcf.xml` (articulated actuators, Goal Placement Pad site, wrist/overhead cameras, end-effector force sensors), `env.py` (`gymnasium.Env` with 29-D observations, 7-D continuous actions, selectable reward presets, and domain randomization), `mjx_env.py` (GPU-vectorized JAX `mujoco.mjx` via `jax.vmap`/`jax.jit`), `train_rl.py` (Stable-Baselines3 PPO/SAC trainer, MJX throughput harness, and calibrated 20-env evaluator), `requirements.txt`, and `README.md`.
- **Synthetic LeRobot v2.1 Episode Dataset Generator (`lerobotDatasetGenerator.ts`):**
  - Rolls out 10–100+ domain-randomized pick-lift-transport-place trajectories through the calibrated rigid-body physics and 6-DOF IK solver (`POST /api/training/lerobot/preview` and `/download`), automatically filtering failed episodes and computing per-feature LeRobot v2.1 normalization statistics (`min`, `max`, `mean`, `std`).
  - Packages `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, per-episode trajectory CSVs (`data/chunk-000/episode_XXXXXX.csv`), `scene.mjcf.xml`, `scripts/build_parquet_and_videos.py` (converts CSVs to native Parquet + renders synchronized MuJoCo wrist/overhead `.mp4` cameras), `scripts/train_lerobot.sh` (tailored for ACT, Diffusion Policy, or SmolVLA), and Hugging Face `README.md` dataset card into a downloadable `.zip`.

### 2.8 QR Judge 6-Second Hand-Demo Retargeting (`MobileCaptureView.tsx`, `mobileSensorCapture.ts`)
- **Tap-to-Select Target Reticle:** Scanning the desktop QR code opens `/capture?session=<id>`, where a judge taps directly on an object in the live rear-camera viewfinder to lock normalized `(tapX, tapY)` coordinates.
- **DeviceOrientation Gravity Vector + Live Microphone Audio:** Samples real `DeviceOrientationEvent` / `DeviceMotionEvent` gravity vectors `(gx, gy, gz)` (with iOS 13+ permission handling and live horizon tilt readout) alongside a live microphone audio track (`audio: true` + Web Audio RMS meter) during a 6.0-second auto-stop recording.
- **Audio-Preserving Analysis Proxy & Carbon Retargeting:** Preserves the 32 kbps mono AAC audio track in `videoTranscoder.ts` (at 2 FPS for short clips) and injects the tap coordinates, averaged gravity vector, and audio cues into Gemini 4 Carbon, which builds the MuJoCo scene, retargets the human hand obstacle-avoidance arc (`pickupPos → apexWaypointPos → placePos`) onto **SO-ARM101** (`sceneMode: 'hand_demo_retarget'`), anchors the 3D Goal Placement Pad to `placePos`, and immediately launches Hydrogen (20 randomized rollouts) + Neon (`xhigh`) to fix slip/collision failures until 20/20 complete the demonstration reliably.

---

## 3. Visual System & Engineering Standards

- **Aesthetic:** Precision Lab Dark (`#07090E` canvas, `#111622` structural surfaces, `#1E293B` 1px hairlines, `#F59E0B` amber primary, `#06B6D4` cyan spatial overlays, `#10B981` pass emerald, `#F43F5E` fault rose).
- **Typography:** `Chakra Petch` (display/headers), `IBM Plex Sans` (UI prose), `JetBrains Mono` (`tabular-nums` telemetry & code).
- **Centralized Configuration:** All model IDs, thinking levels, physics thresholds, upload boundaries, and camera presets live in `src/config/appConfig.ts`.
- **Observability:** All function calls (with parameters) and all GenAI interactions (model, prompt, config, outputs, duration) are logged at `INFO` level via `src/utils/logger.ts` with inline binary data stripped.
