# Sim-to-Real Robotics "Failure-to-Policy" Compiler

## 1. Project Overview & Hackathon Hook

**Target Rubric Alignment (Gemini 4 NYC Hackathon):**
* **Gemini 4 Use (20%):** Distinct, non-overlapping responsibilities across **Gemini 4 Carbon**, **Gemini 4 Hydrogen**, and **Gemini 4 Neon** chained together via the stateful **Interactions API** (`previous_interaction_id`), leveraging `thinking_level` (`low`, `high`, `xhigh`), **95% Implicit Context Caching**, and asynchronous **Long Decode** (`background=True`).
* **Closing the Sim-to-Real Gap (Phone-to-Cloud-Sim):** Robotics teams and hackathon judges can pull out their **smartphone**, scan a QR code on the dashboard, record a 10-second video of a cluttered tabletop or failed robot grasp, and stream it directly to the cloud.
* **The Solution:**
  1. **Mobile Capture & Cloud Ingestion:** A mobile-responsive capture view (`MediaRecorder` / native camera input) uploads raw smartphone video straight to cloud object storage / FastAPI backend.
  2. **Spatial Real-to-Sim Reconstruction:** **Gemini 4 Carbon** infers calibrated 3D object poses, dimensions, surface materials, and failure modes from the phone footage (using standard reference cues like table edges, mugs, or AR fiducials) and compiles a **MuJoCo (`MJCF` XML)** scene.
  3. **Cloud MuJoCo Swarm Execution:** Instead of relying on local laptop CPUs, a serverless **Cloud MuJoCo Worker Swarm** (Modal / Google Cloud Run) executes 20 domain-randomized environments in parallel headless EGL/OSMesa mode and streams rendered video frames back to the browser.
  4. **Closed-Loop Policy Repair:** **Neon** (`xhigh` thinking) inspects contact telemetry from failed cloud rollouts and patches the control policy until simulated pass rates climb from **0% → 100%**.

---

## 2. Gemini 4 Multi-Model & Cloud Architecture

```
[Smartphone Camera (QR Pairing / Direct Mobile Web Capture)]
                 │  (Direct Upload / Chunked Video Stream)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. GEMINI 4 CARBON (thinking_level="high")                  │
│    Role: Mobile Video Spatial & Root-Cause Architect        │
│    • Calibrates 3D tabletop scale, object poses, & obstacle │
│      geometry from handheld smartphone footage.             │
│    • Emits valid MuJoCo MJCF XML + 20 domain-randomized     │
│      parameter variants (friction, mass, pose jitter).      │
└────────────────┬────────────────────────────────────────────┘
                 │ (Passes previous_interaction_id + cached context)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CLOUD MUJOCO WORKER SWARM + GEMINI 4 HYDROGEN (low)      │
│    Role: Cloud Physics Execution & Telemetry Aggregator     │
│    • Cloud workers (Modal / Cloud Run) run 20 headless      │
│      MuJoCo simulations in parallel with EGL rendering.     │
│    • Hydrogen clusters contact forces, joint torques, slip  │
│      velocity, and streams MP4/WebP rollout previews to UI. │
└────────────────┬────────────────────────────────────────────┘
                 │ (Passes failed rollout telemetry via interaction chain)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GEMINI 4 NEON (thinking_level="xhigh")                   │
│    Role: Adversarial Sim-to-Real Debugger & Policy Patcher  │
│    • Audits contact-force spikes, IK singularities, & slip. │
│    • Rewrites trajectory waypoints, grasp approach vectors, │
│      and PD/impedance gains in Python.                      │
│    • Loops with Cloud MuJoCo until Pass Rate = 20/20 (100%).│
└────────────────┬────────────────────────────────────────────┘
                 │ (background=True Long Decode >65K tokens)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. DEPLOYMENT BUNDLE & INTERACTIVE CLOUD REPLAY             │
│    • Verified Python control script (LeRobot / ROS2 ready)  │
│    • Complete MJCF benchmark suite + telemetry audit report │
└─────────────────────────────────────────────────────────────┘
```

### Why This Cannot Run on Legacy Models
1. **Stateful Cross-Model Routing (`previous_interaction_id`):** Carbon's spatial understanding of the smartphone video and MuJoCo world state is inherited directly by Hydrogen and Neon without re-uploading heavy video frames or MJCF XMLs on every repair hop.
2. **Dynamic `thinking_level` Escalation:** Cloud rollout telemetry summarization runs fast and cheap on `gemini-4-hydrogen` (`thinking_level="low"`), while physical root-cause patching escalates to `gemini-4-neon` (`thinking_level="xhigh"`).
3. **95% Implicit Context Caching:** The robot kinematic specs, MJCF scene definitions, and rollout history remain cached across iterative repair loops.
4. **Long Decode (`background=True`):** Emits the full, multi-file deployment package (20 randomized MJCF world files, verified controller code, and diagnostic report) in one uninterrupted generation.

---

## 3. Recommended Tech Stack (24-Hour Build)

* **Mobile Capture Frontend:** Mobile-first web route (`/capture?session=<id>`) with `accept="video/*" capture="environment"` + `MediaRecorder` API, paired via a live QR code on the main desktop dashboard so anyone in the room can record on their phone and watch the cloud simulation spin up on the big screen.
* **Agent Orchestration:** Google Antigravity + `google-genai` Python SDK (Interactions API).
* **Cloud Physics Compute:** **Modal** (`@app.function(cpu=2, memory=4096)` with `.map()` across 20 containers) or **Google Cloud Run Jobs** running headless `mujoco` (`MUJOCO_GL=egl` or `osmesa`) + `imageio[ffmpeg]` to stream rendered MP4/GIF clips back to the UI.
* **Robot Model:** Open-source **SO-ARM101** (`so_arm101.xml` from Hugging Face LeRobot / Google Antigravity Robotics plugin) or Franka Panda (`franka_emika_panda` from `mujoco_menagerie`).
* **Dashboard UI:** FastAPI (with WebSockets / SSE for live phone-upload notifications and iteration updates) + single-file Tailwind/Vanilla JS dashboard displaying:
  * **QR Code Phone Pairer** + Uploaded Smartphone Video Player.
  * **Cloud MuJoCo 3D Viewport** (live rendered rollout videos from cloud workers).
  * **Live Closed-Loop Counter:** `Cloud Rollouts Passing: 2/20 → 11/20 → 20/20 (100%)`.

---

## 4. Directory Structure for Antigravity

```
sim2real-compiler/
├── app/
│   ├── main.py                  # FastAPI server, mobile upload endpoint, QR session & SSE stream
│   ├── orchestrator.py          # Gemini 4 Interactions API state machine
│   ├── prompts.py               # Spatial calibration & policy-patching prompts
│   └── schemas.py               # Pydantic schemas for MJCF configs & rollout telemetry
├── cloud_sim/
│   ├── modal_worker.py          # Serverless cloud MuJoCo swarm (20 parallel containers + EGL render)
│   ├── assets/                  # SO-ARM101 / Panda MJCF & mesh assets
│   └── ik_controller.py         # Baseline Jacobian IK + impedance controller (patched by Neon)
├── frontend/
│   ├── index.html               # Desktop command center (QR code, cloud sim player, telemetry bar)
│   └── mobile_capture.html      # Phone-friendly camera recorder & direct upload page
├── requirements.txt
└── .env                         # GEMINI_API_KEY=... & MODAL_TOKEN_ID=...
```

---

## 5. Cloud MuJoCo Swarm Worker (`cloud_sim/modal_worker.py`)

Using **Modal** (zero-config serverless Python cloud compute) lets you fan out all 20 MuJoCo physics simulations across 20 cloud containers in ~3 seconds while rendering video frames headlessly via `osmesa`/`egl`:

```python
import os
import base64
import modal

# Define cloud container image with MuJoCo + headless rendering
sim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libosmesa6-dev", "libgl1-mesa-glx", "libglfw3", "ffmpeg")
    .pip_install("mujoco>=3.2.0", "numpy", "imageio[ffmpeg]")
)

app = modal.App("gemini4-sim2real-swarm")


@app.function(image=sim_image, cpu=2, memory=2048, timeout=120)
def evaluate_cloud_env(mjcf_xml: str, rand_params: dict, controller_code: str, render_video: bool = False) -> dict:
    os.environ["MUJOCO_GL"] = "osmesa"
    import mujoco
    import numpy as np
    import imageio.v3 as iio

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    # Apply domain randomization (friction, mass, XY pose jitter)
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object")
    if target_geom_id != -1:
        model.geom_friction[target_geom_id, 0] = rand_params.get("friction", 0.5)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
    if target_body_id != -1:
        model.body_mass[target_body_id] = rand_params.get("mass", 0.15)
        model.body_pos[target_body_id, :2] += np.array(rand_params.get("xy_offset", [0.0, 0.0]))

    mujoco.mj_forward(model, data)

    # Load generated controller into isolated namespace
    local_ns = {"np": np, "mujoco": mujoco}
    exec(controller_code, local_ns)
    execute_policy = local_ns["execute_policy"]

    renderer = mujoco.Renderer(model, height=360, width=480) if render_video else None
    frames = []
    max_contact_force = 0.0

    for step in range(1500):  # 3.0s at 500Hz
        execute_policy(model, data, step)
        mujoco.mj_step(model, data)

        for i in range(data.ncon):
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            max_contact_force = max(max_contact_force, float(np.linalg.norm(c_force[:3])))

        # Capture 25 FPS frames for UI replay on representative environments
        if renderer is not None and step % 20 == 0:
            renderer.update_scene(data)
            frames.append(renderer.render())

    final_z = float(data.xpos[target_body_id, 2]) if target_body_id != -1 else 0.0
    success = (final_z > 0.12) and (max_contact_force < 45.0)

    video_b64 = None
    if frames:
        mp4_bytes = iio.imwrite("<bytes>", frames, extension=".mp4", fps=25)
        video_b64 = base64.b64encode(mp4_bytes).decode("utf-8")

    return {
        "env_id": rand_params["id"],
        "success": success,
        "final_object_z": round(final_z, 4),
        "peak_contact_force_n": round(max_contact_force, 2),
        "params": rand_params,
        "video_b64": video_b64,
    }


async def run_cloud_swarm(mjcf_xml: str, randomizations: list, controller_code: str) -> dict:
    """Fans out 20 parallel MuJoCo rollouts across Modal cloud containers."""
    tasks = [
        (mjcf_xml, r, controller_code, idx < 2)  # Render MP4 previews for first 2 envs
        for idx, r in enumerate(randomizations)
    ]
    results = []
    async for res in evaluate_cloud_env.starmap.aio(tasks):
        results.append(res)

    passed = sum(1 for r in results if r["success"])
    preview_videos = [r.pop("video_b64") for r in results if r.get("video_b64")]
    return {
        "passed": passed,
        "total": len(results),
        "preview_videos_b64": preview_videos,
        "rollouts": results,
    }
```

---

## 6. Gemini 4 Interactions API Orchestrator (`app/orchestrator.py`)

```python
import os
import json
import asyncio
from google import genai
from google.genai import types
from cloud_sim.modal_worker import run_cloud_swarm

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


async def run_failure_to_policy_pipeline(
    video_path: str,
    task_goal: str,
    event_callback=None,
    max_iterations: int = 5,
):
    """
    Executes the closed-loop Gemini 4 Sim-to-Real pipeline from a mobile phone recording:
    1. Carbon: Phone Video -> Scale-Calibrated MuJoCo MJCF + Domain Randomization Spec
    2. Cloud MuJoCo Swarm + Hydrogen: Parallel Cloud Rollouts & Telemetry Clustering
    3. Neon: xhigh Thinking Policy Patching Loop
    4. Long Decode: Full Deployment Bundle Generation
    """
    async def emit(event_type: str, payload: dict):
        if event_callback:
            await event_callback({"type": event_type, **payload})

    await emit("status", {"stage": "uploading", "message": "Uploading smartphone video to Gemini Files API..."})
    video_file = client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        await asyncio.sleep(2)
        video_file = client.files.get(name=video_file.name)

    # ---------------------------------------------------------
    # STEP 1: GEMINI 4 CARBON — Mobile Video Spatial Reconstruction
    # ---------------------------------------------------------
    await emit("model_active", {"model": "gemini-4-carbon", "thinking_level": "high", "step": "Spatial Real-to-Sim Reconstruction"})
    carbon_turn = client.interactions.create(
        model="gemini-4-carbon",
        input=[
            video_file,
            (
                f"Analyze this handheld smartphone video showing a real-world tabletop/robot setup for task: '{task_goal}'. "
                "1) Estimate real-world metric scale (using standard tabletop objects like mugs, hands, or robot links) "
                "and reconstruct 3D object poses, bounding dimensions, and surface friction estimates. "
                "2) Identify any physical failure mode or obstacle hazards visible in the video. "
                "3) Generate a valid MuJoCo MJCF XML scene (`target_body` + `target_object` geom + robot arm) reproducing the scene. "
                "4) Output a JSON object with keys: `failure_analysis`, `mjcf_xml`, `initial_controller_python`, "
                "and `randomizations` (20 dicts with `id`, `friction`, `mass`, `xy_offset`)."
            ),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            response_mime_type="application/json",
        ),
        store=True,
    )

    scene_spec = json.loads(carbon_turn.outputs[-1].text)
    last_interaction_id = carbon_turn.id
    current_policy_code = scene_spec["initial_controller_python"]
    await emit("scene_ready", {"failure_analysis": scene_spec["failure_analysis"]})

    # ---------------------------------------------------------
    # STEP 2 & 3: CLOUD MUJOCO + HYDROGEN <-> NEON (xhigh Patcher)
    # ---------------------------------------------------------
    for iteration in range(1, max_iterations + 1):
        await emit("cloud_sim", {"iteration": iteration, "message": "Running 20 parallel MuJoCo environments in the cloud..."})
        raw_sim_results = await run_cloud_swarm(
            mjcf_xml=scene_spec["mjcf_xml"],
            randomizations=scene_spec["randomizations"],
            controller_code=current_policy_code,
        )

        # Stream cloud simulation videos + pass rate immediately to UI
        await emit("sim_results", {
            "iteration": iteration,
            "passed": raw_sim_results["passed"],
            "total": raw_sim_results["total"],
            "preview_videos_b64": raw_sim_results["preview_videos_b64"],
        })

        await emit("model_active", {"model": "gemini-4-hydrogen", "thinking_level": "low", "step": f"Iteration {iteration} Telemetry Triage"})
        hydrogen_turn = client.interactions.create(
            model="gemini-4-hydrogen",
            previous_interaction_id=last_interaction_id,
            input=(
                f"Iteration {iteration} Cloud MuJoCo Rollout Telemetry:\n"
                f"{json.dumps(raw_sim_results['rollouts'], indent=2)}\n\n"
                "Cluster the failure modes across the 20 cloud environments, compute the exact pass rate, "
                "and highlight peak contact-force anomalies or slip velocities."
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
            store=True,
        )
        last_interaction_id = hydrogen_turn.id
        telemetry_summary = hydrogen_turn.outputs[-1].text

        if raw_sim_results["passed"] == raw_sim_results["total"]:
            break

        # Escalate to Gemini 4 Neon (xhigh thinking) to patch the control policy
        await emit("model_active", {"model": "gemini-4-neon", "thinking_level": "xhigh", "step": f"Iteration {iteration} Policy Repair"})
        neon_turn = client.interactions.create(
            model="gemini-4-neon",
            previous_interaction_id=last_interaction_id,
            input=(
                f"Telemetry Diagnosis:\n{telemetry_summary}\n\n"
                "Rewrite `execute_policy(model, data, step)` in Python to fix every failure mode "
                "(adjust pre-grasp approach offset, impedance stiffness, grip force ramp, or waypoint timing). "
                "Return ONLY valid JSON with keys: `root_cause_fix` and `patched_controller_python`."
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="xhigh"),
                response_mime_type="application/json",
            ),
            store=True,
        )
        last_interaction_id = neon_turn.id
        patch_payload = json.loads(neon_turn.outputs[-1].text)
        current_policy_code = patch_payload["patched_controller_python"]
        await emit("policy_patched", {"iteration": iteration, "fix": patch_payload["root_cause_fix"]})

    # ---------------------------------------------------------
    # STEP 4: LONG DECODE — Full Deployment & Audit Package
    # ---------------------------------------------------------
    await emit("model_active", {"model": "gemini-4-neon", "thinking_level": "high", "step": "Long Decode Bundle Compilation"})
    final_bundle = client.interactions.create(
        model="gemini-4-neon",
        previous_interaction_id=last_interaction_id,
        input=(
            "Generate the complete, production-ready deployment package containing: "
            "1) Standalone LeRobot / MuJoCo controller script with safety torque limits, "
            "2) All 20 verified MJCF regression test scenes, and "
            "3) A comprehensive Sim-to-Real Verification & Contact Dynamics Audit Report."
        ),
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            max_output_tokens=131072,
        ),
        background=True,
        store=True,
    )

    await emit("complete", {"final_interaction_id": final_bundle.id})
    return {
        "final_interaction_id": final_bundle.id,
        "verified_policy": current_policy_code,
        "mjcf_xml": scene_spec["mjcf_xml"],
    }
```

---

## 7. Mobile Capture & FastAPI Session Bridge (`app/main.py`)

To make smartphone recording effortless during a live hackathon demo:
1. The desktop dashboard (`frontend/index.html`) generates a **QR Code** pointing to `https://<your-ngrok-or-cloud-url>/capture?session=<id>`.
2. Scanning the QR code opens `frontend/mobile_capture.html` on iOS Safari / Android Chrome, which triggers the native rear camera (`<input type="file" accept="video/*" capture="environment">`) and `POST`s the recording directly to `/api/sessions/{session_id}/upload`.
3. FastAPI immediately triggers `run_failure_to_policy_pipeline` in the background and streams progress events + rendered cloud MuJoCo MP4 clips over Server-Sent Events (`/api/sessions/{session_id}/events`) to the desktop screen.

---

## 8. Antigravity Kickoff Prompt

Paste this directly into **Google Antigravity** once you open your workspace folder with this Markdown file:

> *"Read `sim2real_robotics_blueprint.md` and scaffold the complete project. Download the open-source SO-ARM101 MuJoCo MJCF model into `cloud_sim/assets/`, implement `cloud_sim/modal_worker.py` for parallel cloud MuJoCo execution with headless video rendering, implement `app/orchestrator.py` and `app/main.py` with QR-paired smartphone video upload and SSE live telemetry, and build both `frontend/mobile_capture.html` (phone camera recorder) and `frontend/index.html` (live command center showing the phone video, cloud MuJoCo rollout videos, Gemini 4 model-routing bar, and pass-rate counter)."*