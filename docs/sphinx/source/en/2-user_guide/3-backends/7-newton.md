# Newton Backend

[Newton](https://github.com/newton-physics/newton) (PyPI distribution
`newton`, pinned to 1.5.1) is a GPU physics simulator built on Warp that
UniLab runs **in-process**: `unisim.backend.newton.NewtonBackend` serves the
standard `SimBackend` NumPy contract on top of it, so physics shares the
training process with the learner — no worker subprocess, no IPC.

Current status: Newton is wired at the owner/config boundary (UniLab PR
[#1511](https://github.com/unilabsim/UniLab/pull/1511); the adapter lives in
the unisim repository as `unisim.backend.newton`). `g1_walk_flat` ships PPO
and SAC owner configs
(`src/unilab/conf/{ppo,sac}/task/g1_walk_flat/newton.yaml`, with `G1WalkFlat`
registered for the `newton` backend), and the cross-backend contract audit
(`scripts/audit_sim2sim_contracts.py`) covers the mujoco/newton pair in both
algo trees (verdict TRANSFERABLE). Support levels: SAC is **Tested** — a
full 5000-iteration training completed on 2026-09-06 on an RTX 4090 /
torch 2.8.0+cu128 / newton 1.5.1 / mujoco-warp 3.11 (reward/mean 6.68 →
242.3, episode length → 983, ~43k steps/s, 4m25s wall), with playback
validated on `model_5000.pt`: native ViewerGL offscreen record (800-frame
1280x720 mp4) and an interactive ViewerGL window smoke on a live X
display; PPO remains **Configured**
(evidence limited to the owner configs, compose/contract checks, and
fail-closed runtime/import boundaries; no training or playback claim yet).

Multi-GPU data parallelism (issue
[#1512](https://github.com/unilabsim/UniLab/issues/1512)): verified on
2026-09-06 on 2x NVIDIA RTX 6000D (Blackwell) / torch 2.8.0+cu128 /
newton 1.5.1 / mujoco-warp 3.11 — PPO torchrun DP=2 and SAC
DpRankSupervisor DP=2 (`training.devices=[0,1]`) training smokes both
complete, and `nvidia-smi` sampling confirms each rank's learner and
collector sim processes land on their own physical GPU with no cross-GPU
leakage; single-GPU PPO/SAC regressions pass alongside. Newton/Warp follows
standard CUDA device semantics, so no `CUDA_VISIBLE_DEVICES` pinning (the
Genesis quirk) is needed; the rank-local device reaches spawn collectors as
a `newton_device="cuda:N"` env override, and uni_rl's collector-side
process binding is injection-based — UniLab injects
`bind_backend_process_device_for_backend`, which covers both mjwarp and
newton — while the SAC owner raises the
collector tick-0 timeout to 180 s to cover Warp kernel compilation on the
cold path.

## Installation

The Newton runtime is an optional extra, pinning Newton 1.5.1 with the
MuJoCo-Warp 3.11 / Warp 1.16 line:

```bash
# In a source checkout:
uv sync --extra newton

# From PyPI:
pip install "unilab[newton]"
```

The extra pins `newton==1.5.1`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`, and
`warp-lang==1.16.0` exactly and includes the native ViewerGL dependencies
(`pyglet>=2.1.6,<3`, `imgui-bundle>=1.92.0`). These sit on the same MuJoCo 3.11 /
MuJoCo-Warp 3.11 / Warp 1.16 line as the `mujoco` extra (`mujoco~=3.11.0`)
and the `mjwarp` extra (`mujoco-warp~=3.11.0`, `warp-lang==1.16.0`), so all
three extras are **jointly installable** in one environment:

```bash
uv sync --extra mujoco --extra mjwarp --extra newton
```

Prerequisites:

- Linux with an NVIDIA GPU and CUDA driver. The adapter's process device
  binding (`unisim.backend.newton.runtime.bind_newton_process_device`)
  requires an active CUDA Warp device and fails closed otherwise; a CPU
  device is not a validated support lane.
- Python `>=3.10` (the repository supports 3.10–3.13).

After installation, the unisim repository provides
`scripts/check_newton_runtime.py` as a metadata-only probe (pass `--import`
to import the native runtime explicitly). On the UniLab side a missing Newton
runtime is not silent: the top-level CLI checks the `newton`, `mujoco_warp`,
`mujoco`, and `warp` modules before training and fails closed with an
install hint (`_check_runtime_requirements` in `src/unilab/cli.py`).

## Training and Evaluation

Training selects the newton owner through the canonical CLI:

```bash
# PPO
uv run train --algo ppo --task g1_walk_flat --sim newton

# SAC
uv run train --algo sac --task g1_walk_flat --sim newton
```

Newton/MuJoCo-Warp 3.11 owns explicit device and storage capacities, exposed
as `env.*` fields in the owner YAML:

- `newton_device`: the owner's `null` default is intentional — the process
  device binder (`src/unilab/base/process_device.py`) injects the rank-local
  CUDA device before materialization. An explicit value must be a non-empty
  CUDA device string.
- `newton_nconmax` / `newton_njmax`: explicit capacity bounds (320 / 512 in
  the g1 owner). The adapter calibrates solver counts on the cold path and
  raises an explicit capacity error when a bound is too small; it never
  silently truncates constraints.
- `newton_capacity_check_steps`: how often capacity is checked (default 1).

## Playback and Rendering

The Newton backend renders natively through the upstream
`newton.viewer.ViewerGL` (included in the `newton` extra,
`pyglet>=2.1.6,<3` + `imgui-bundle>=1.92.0`). The owner inherits the base config's
`training.play_render_mode: auto`: with a display, `auto` resolves to
`interactive` (the ViewerGL window); without one it resolves to `record`
(`ViewerGL(headless=True)` offscreen rendering to mp4). If an installation is
incomplete, `record` falls back to the MuJoCo offline snapshot renderer while
`interactive` fails closed; a normal `newton` install always selects the
native renderer. Headless offscreen rendering still needs an OpenGL
context: set `PYOPENGL_PLATFORM=egl` on display-less Linux hosts and
`PYOPENGL_PLATFORM=glx` under Wayland.

```bash
uv run eval --algo ppo --task g1_walk_flat --sim newton \
    --load-run <run_dir_name> --render-mode record

uv run eval --algo sac --task g1_walk_flat --sim newton \
    --load-run <run_dir_name> --render-mode record
```

## Unsupported Boundaries

The following fail closed (explicit error or rejected configuration) rather
than silently degrading:

- **Runtime PD-gain randomization**: Newton currently rejects it, and the
  owner keeps `events.pd_gains: null` to stay fail-closed until the adapter
  adds the capability.
- **Camera kwargs on the native render path**: the native ViewerGL path
  ignores `camera_kwargs` (the MuJoCo snapshot path still honors them).

## Cross-Backend Migration (sim2sim)

The newton owner keeps DENYLIST parity with the MuJoCo owner under the audit
guard (`src/unilab/utils/sim2sim.py`, verdict TRANSFERABLE), so checkpoints
of the same task transfer across backends.
