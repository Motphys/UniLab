# Genesis Backend

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) (PyPI distribution
`genesis-world`, pinned to 1.3.3) is a GPU physics simulator that UniLab runs
**in-process**: `GenesisBackend` serves the standard `SimBackend` NumPy
contract on top of it, so physics shares the training process with the
learner — no worker subprocess, no IPC.

Current status: `GenesisBackend` is implemented and registered; `g1_walk_flat`
ships PPO and SAC owner configs
(`src/unilab/conf/{ppo,sac}/task/g1_walk_flat/genesis.yaml`), and the cross-backend
contract audit (`scripts/audit_sim2sim_contracts.py`) covers the
mujoco/genesis pair in both algo trees (verdict TRANSFERABLE). Support level
is **experimental**. Evidence: registry + owner YAML + compose/contract
coverage, plus a real-machine slow-lane env smoke
(`tests/envs/locomotion/g1/test_g1_owner_contract.py`: compose -> env
construction -> keyframe reset -> 12 finite steps -> cleanup, run for both
the ppo and sac trees). The SAC cell is marked `Tested` after a full
real-machine training validation (5000/5000 iterations on 2026-08-31, RTX
4090 / torch 2.8.0+cu128 / genesis-world 1.3.3: reward/mean 6.5 -> 244.8,
episode length -> 987/1000, 10.26M env steps in 224 s wall time) plus record
playback validation on the final checkpoint; the PPO cell stays `Configured`
(no training validation yet).

Env-construction lifecycle (fixed in #1383): entity validation during
`ManagerBasedRlEnv` construction reads state getters before the env's
`materialize()` hook, so the adapter's `materialize()` is idempotent and
lazily triggered (the first state access completes `scene.build`, the same
pattern as the IsaacGym backend). The adapter design otherwise follows the
measured mappings of `scripts/tools/genesis_feasibility/REPORT.md`.

## Model Contract

Genesis 1.3.3 drops three MJCF features at import (REPORT §3): the global
`<option>` block, `<keyframe>`, and the whole `<sensor>` block. The adapter
compensates on the materialize cold path — hot paths never parse XML:

- **Global option**: the owner YAML re-declares what the MJCF `<option>`
  carried. For g1 (`<option integrator="implicitfast"
  timestep="0.006666666666666667"/>`) the timestep flows through the existing
  `sim_dt` / `ctrl_dt` chain and the integrator through
  `env.genesis_integrator: implicitfast`. Constraint solver, friction cone,
  and solver iterations stay at Genesis defaults (`env.genesis_*` fields are
  `None`): the MJCF option block does not declare them and MuJoCo's implicit
  defaults (PGS solver, pyramidal cone) have no Genesis equivalent.
- **Keyframes**: scanned once with the `mujoco` package at materialize and
  cached, so `default_keyframe_name: stand` resets work unchanged.
- **Sensors**: MJCF-named equivalents computed from link state, plus one
  `IMUSensor` per accelerometer site with clean (noise-free) data. Contact
  sensors with `data="found"` become a per-link net-contact-force threshold
  (1 N) — an approximation of the geom-pair `found` semantic, not a
  reproduction of it.
- **Actuators**: `<position kp kv>` actuators import losslessly and are
  driven with `control_dofs_position`; kp/kd reset randomization is supported
  (the DR capability set declares only the per-env round-trip-measured terms:
  body mass, base mass, kp, kd, plus interval body force).

## Prerequisites

- Linux x86_64 with an NVIDIA GPU and driver. Only the `gs.gpu` lane is
  validated by the feasibility probe; the CPU backend is not a validated
  support lane.
- The repository's pinned torch window (torch 2.8 on x86_64): the
  IMUSensor + contact-force sensor combination crashed under torch 2.7 in
  the probe (REPORT §3.4/§8).
- Install the optional extra (pins `genesis-world==1.3.3` exactly):

```bash
uv sync --extra genesis
```

## Training and Evaluation

Training selects the genesis owner through the canonical CLI (owner YAML
compose, registry routing, and env construction are covered by the
real-machine slow-lane smoke; there is no training-convergence evidence yet):

```bash
# PPO
uv run train --algo ppo --task g1_walk_flat --sim genesis

# SAC
uv run train --algo sac --task g1_walk_flat --sim genesis

# Small smoke run: 64 environments, 3 iterations only
uv run train --algo ppo --task g1_walk_flat --sim genesis \
    algo.num_envs=64 algo.max_iterations=3
```

## Playback and Rendering

Genesis native rendering is a declared capability and attaches lazily after
`scene.build` (no viewer dependency on the training hot path): the
interactive viewer is a post-build `genesis.vis.viewer.Viewer`, and recording
uses a post-build offscreen camera. The owner sets
`training.play_render_mode: auto`, which resolves per host:

- `auto`: opens the **interactive viewer** when a display is reachable
  (`DISPLAY`/`WAYLAND_DISPLAY`), otherwise falls back to offscreen
  **recording**.
- `interactive`: always opens the viewer; on a headless host
  `init_renderer` fails with an actionable error. Closing the viewer window
  ends playback cleanly (the backend raises `RenderClosedError`, which the
  play loop treats as a normal exit).
- `record`: headless offscreen capture, requires finite
  `training.play_steps` and an output path; frames are written to
  `play_video.mp4` in the checkpoint's run directory at `1/ctrl_dt` fps.
- `none`: playback is skipped safely as a no-op.

The camera follows the repo-wide spherical kwargs (`cam_distance` /
`cam_elevation` / `cam_azimuth`, plus `cam_tracking` to follow the root of
`cam_tracking_env_idx`). Evaluate a trained run with:

```bash
uv run eval --algo sac --task g1_walk_flat --sim genesis \
    --load-run <run_dir_name> --render-mode record
```

`get_physics_state` snapshots remain undeclared: playback drives the live
scene, so state-snapshot playback is not needed.

## Lifecycle: One `gs.init` Per Process

Genesis init/destroy cycles leak 200–450 MB of host RSS per cycle (REPORT
§3.5 [9a]), so the adapter allows exactly **one `gs.init` per process**:
after the session is destroyed, constructing another backend fails closed
with an explanatory error. One training run per process satisfies this
constraint by design; do not build genesis envs repeatedly inside a
long-lived host process.

## Unsupported Boundaries

The following fail closed with explicit errors rather than silently
degrading:

- **Geom-name contract** (`get_geom_names` and friends): not exposed.
- **Generated terrain and height scanners**: `scene.terrain` is rejected;
  select a flat owner YAML.
- **Absolute geom-friction DR**: upstream Genesis only offers a per-env
  friction *ratio* API, so `geom_friction` reset randomization and
  `get_geom_friction` fail closed.
- **Interval push / body-velocity-delta DR**: rejected at construction/plan
  time (`push_body_name`, push perturbation).
- **Repeated `gs.init`** in one process (see above).

## Cross-Backend Migration (sim2sim)

The genesis owner keeps DENYLIST parity with the MuJoCo owner under the audit
guard (`src/unilab/utils/sim2sim.py`, verdict TRANSFERABLE), so checkpoints of
the same task transfer across backends. Native rendering exists on both
sides, but the implementations differ: MuJoCo renders offline from physics
state snapshots, while Genesis drives its live scene with a post-build
viewer/camera.
