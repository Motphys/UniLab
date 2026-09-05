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
242.3, episode length → 983, ~43k steps/s, 4m25s wall), with record
playback validated on `model_5000.pt`; PPO remains **Configured**
(evidence limited to the owner configs, compose/contract checks, and
fail-closed runtime/import boundaries; no training or playback claim yet).

## Installation

The Newton runtime is an **isolated optional extra**, pinning Newton 1.5.1
with the MuJoCo-Warp 3.11 / Warp 1.16 line:

```bash
# In a source checkout:
uv sync --extra newton

# From PyPI:
pip install "unilab[newton]"
```

The extra pins `newton==1.5.1`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`, and
`warp-lang==1.16.0` exactly. That is a different MuJoCo version line from the
historical `mjwarp` extra (`mujoco-warp==3.10.0.3`) and the `mujoco` extra,
so `pyproject.toml` declares them **mutually exclusive** through uv
`conflicts`: Newton cannot share one environment with the `mujoco` or
`mjwarp` extras. To compare MuJoCo / MJWarp against Newton, keep Newton in a
separate environment.

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

The Newton backend has no native UniLab viewer yet: the owner sets
`training.play_render_mode: record`, so playback takes the record/headless
path, and requesting a native interactive renderer fails closed.

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
- **Native interactive rendering**: see Playback above.

## Cross-Backend Migration (sim2sim)

The newton owner keeps DENYLIST parity with the MuJoCo owner under the audit
guard (`src/unilab/utils/sim2sim.py`, verdict TRANSFERABLE), so checkpoints
of the same task transfer across backends.
