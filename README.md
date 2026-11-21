# LetThereBeSims

**LetThereBeSims** is a robotics failure-to-policy workbench powered by the **Gemini 4 Interactions API**, Three.js/WebGL, and MuJoCo-compatible MJCF compilation. Upload 1 or 2 camera views of a tabletop manipulation run (failure, successful demo, or robot-free workspace) or pair a smartphone via QR code to calibrate a metric 3D scene, run a 20-environment domain-randomized verification swarm, iteratively repair Python control policies, and export a deployment bundle.

## Features

- **1- or 2-View Video Intake & QR Mobile Pairing:** Upload a single video (Single View) or two synchronized camera angles (unlocking Dual 3D View), record via desktop webcam, or scan the live QR code on a smartphone (`/capture?session=<id>`).
- **Automatic Outcome & Scene Detection (Gemini 4 Carbon):** Automatically classifies footage as **Failure (Policy Repair)**, **Success (Robustness Stress Test)**, or **No Robot (Simulation-Ready 3D Scene + 2D Spatial Map)**, with automatic 1 FPS / 480p `ffmpeg` analysis proxy compression for long clips.
- **20-Environment Verification Swarm & Policy Patching:** Evaluates 20 domain-randomized environments via the built-in server physics evaluator or an external Cloud MuJoCo webhook, triages anomalies with **Gemini 4 Hydrogen**, patches controllers with **Gemini 4 Neon**, and compiles an async Long Decode deployment bundle (`.py`, `.xml`, `.md`).
- **Dynamic 3D Robot Simulation Stage:** Interactive Three.js workbench for `SO-ARM101` (6-DOF) and `Franka Panda` (7-DOF) featuring live physics & 6-DOF IK, direct 3D object drag-and-drop, configurable rollout lengths (`Full Clip`, `15s`, `30s`, `60s`), and in-browser `.webm` 3D clip recording.

## Prerequisites

- **Node.js** 20+
- **ffmpeg** (optional but recommended on `PATH` for automatic 1 FPS / 480p proxy compression of large video uploads)
- A **Gemini API key** (`GEMINI_API_KEY`)

## Local Setup

1. Clone the repository and install dependencies:

   ```bash
   npm install
   ```

2. Create a `.env` file from `.env.example`:

   ```bash
   cp .env.example .env
   ```

   Set `GEMINI_API_KEY` inside `.env` (never commit `.env`).

3. Start the full-stack dev server on port `3000`:

   ```bash
   npm run dev
   ```

4. Open `http://localhost:3000` in your browser.

## Production Build

```bash
npm run build
npm start
```

## Architecture & Design

See [Design.md](./Design.md) for full architecture, Gemini 4 stage chaining (`previous_interaction_id`), and telemetry specifications.
