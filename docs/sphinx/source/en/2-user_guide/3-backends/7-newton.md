# Newton Backend


[Newton](https://github.com/newton-physics/newton) (PyPI distribution
`newton`, pinned to 1.5.1) is a GPU physics simulator built on Warp that
UniLab runs **in-process**: `unisim.backend.newton.NewtonBackend` serves the
tensor lane of the public `SimBackend` contract on top of it, so physics
shares the training process with the learner — no worker subprocess, no IPC.

Current status: issue #2048 restores Newton to the tensor-only Manager runtime
for exactly two canonical workloads:

```bash
uv run train --algo sac --task g1_walk_flat --sim newton
uv run train --algo flashsac --task g1_motion_tracking --sim newton
```

The owner configurations are
`src/unilab/conf/sac/task/g1_walk_flat/newton.yaml` and
`src/unilab/conf/flashsac/task/g1_motion_tracking/newton.yaml`. Other Newton
owners remain out of scope and do not constitute production support claims.
The public tensor lane exposes device-resident state views, frame/contact
sensors, state widths, tensor stepping, and authoritative selected-reset
publication. Reset randomization that Newton does not support stays explicitly
disabled in the owners; it never falls back to a host composer.

Newton/Warp follows standard CUDA device semantics, so single-GPU runs do not
need `CUDA_VISIBLE_DEVICES` pinning. In multi-GPU topology, the rank-local
device reaches spawn collectors as a `newton_device="cuda:N"` override, and
uni_rl's collector process binding is injection-based; UniLab injects
`bind_backend_process_device_for_backend`, which covers MJWarp and Newton.

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
runtime is not silent: the top-level CLI checks the `newton` module before
training and fails closed with an install hint
(`_check_runtime_requirements` in `src/unilab/cli.py`).

## Training and Evaluation

The canonical tensor-runtime commands above select the Newton owner. Other
historical owner paths remain out of scope until separately enabled.

Newton/MuJoCo-Warp 3.11 owns explicit device and storage capacities, exposed
as `env.*` fields in the owner YAML:

- `newton_device`: explicit backend CUDA placement. The single-device owners
  use `cuda:0`; multi-GPU training overrides this cold-path field from rank
  topology.
- `newton_nconmax` / `newton_njmax`: explicit capacity bounds (320 / 512 in
  the canonical G1 owners). The adapter calibrates solver counts on the cold
  path and raises an explicit capacity error when a bound is too small; it
  never silently truncates constraints.
- `newton_capacity_check_steps`: how often capacity is checked (default 1).
- `newton_use_cuda_graph`: CUDA graph replay (default `true`). A graph-capable
  UniSim runtime captures both Newton state parities after cold capacity
  calibration; ineligible CUDA drivers or capture failures warn and fall back
  to eager execution.
  Runtime evidence must report the diagnostic's actual `enabled` state rather
  than infer it from the request.

## Playback and Rendering

The canonical owners select `training.play_render_mode: record`. Newton renders
through the upstream `newton.viewer.ViewerGL` included in the `newton` extra;
if that installation is incomplete, record falls back to the MuJoCo snapshot
renderer. Headless offscreen rendering still needs an OpenGL context: set
`PYOPENGL_PLATFORM=egl` on display-less Linux hosts and
`PYOPENGL_PLATFORM=glx` under Wayland.

```bash
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
