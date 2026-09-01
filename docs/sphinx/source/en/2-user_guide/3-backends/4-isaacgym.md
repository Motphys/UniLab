# IsaacGym Backend

IsaacGym (NVIDIA Preview 4) is an end-of-life GPU physics simulator from NVIDIA
that only supports Python 3.6-3.8. The UniLab main environment requires
Python >= 3.10, so IsaacGym cannot be installed into it; it is used through an
external, standalone Python 3.8 environment located purely via environment
variables, with no machine-local paths written into the repository.

Current status: `IsaacGymBackend` (a subprocess backend whose physics runs in
the external Python 3.8 worker) is implemented and registered; `g1_walk_flat`
ships isaacgym owner configs
(`src/unilab/conf/{ppo,sac}/task/g1_walk_flat/isaacgym.yaml`), and the cross-backend
contract audit (`scripts/audit_sim2sim_contracts.py`) covers the
mujoco/isaacgym pair. Playback rendering uses IsaacGym's native rendering
(viewer + camera sensor); both interactive and video-recording modes work
(see "Training and Evaluation" below). Real-machine end-to-end validation
depends on the external environment described below and is not covered by
repo CI. The repository also ships a physics benchmark script
`scripts/benchmark/physics/benchmark_physics_step_isaacgym.py`, which locates
the external environment through variables such as
`UNILAB_BENCHMARK_HOLOSOMA_DEPS`. This page covers preparing the external
environment, training and evaluation, benchmark validation, and
troubleshooting.

## Model Contract

The backend consumes the task's MJCF scene directly, but IsaacGym's MJCF
importer is only partially trusted: kinematics (body/dof names and order) are
verified against a host-side XML scan at INIT, and every actuation-relevant
parameter is parsed from the XML rather than read from the importer.

- **Control**: only `<position kp kv forcerange>` actuators are supported —
  `SimBackend.step(ctrl)` carries per-DoF position targets, reproduced with
  PhysX `DOF_MODE_POS` drives (force = kp·(target − q) − kv·q̇, clamped to the
  symmetric forcerange). Scenes with `<motor>`/`<velocity>`/other actuator
  types, non-unit gear, or asymmetric forceranges fail closed at scene scan.
- **Self-collision is disabled** (actor collision filter). MJCF
  `<contact><exclude>` pairs (e.g. G1's elbow↔wrist and pelvis↔hip overlaps)
  cannot be reproduced per link pair through the gymapi; disabling
  self-collision entirely is the ecosystem-standard approximation and a
  superset of the exclusions. Models that rely on self-contact are not
  faithfully reproduced.
- **Joint limits**: the importer drops them, so PhysX applies no joint stops;
  `get_joint_range()` still reports the XML values. Joint `armature` and
  `frictionloss` (resolved through MJCF default classes) are applied to the
  PhysX dofs.

## Prerequisites

- Linux x86_64 with an NVIDIA GPU driver installed.
- Network access to the NVIDIA download site: the script downloads
  `IsaacGym_Preview_4_Package.tar.gz` automatically from
  <https://developer.nvidia.com/isaac-gym-preview-4> (no login required). On
  offline machines, download it yourself first and pass `--tarball <path>`.
- Disk space: roughly 5 GB for miniconda, the conda environment, and the
  IsaacGym package combined.

## Automated Setup

From the repository root, run:

```bash
scripts/tools/setup_isaacgym_env.sh
```

The script installs everything under `$HOME/.unilab/isaacgym` by default;
override the install root with the `UNILAB_ISAACGYM_HOME` environment variable.
The tarball is downloaded automatically to
`$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz`; on offline machines,
pass a pre-downloaded package with `--tarball <path>`. The script is
idempotent and skips completed steps when re-run.

The setup flow: a dedicated miniconda, then a Python 3.8 `hsgym` conda
environment (including `libstdcxx-ng`, which fixes the GLIBCXX issue on Ubuntu
24.04), then unpacking the tarball, `pip install -e isaacgym/python`, and
finally an import self-check.

After installation, add the export lines printed by the script to your shell rc
(e.g. `~/.bashrc`):

```bash
export UNILAB_BENCHMARK_HOLOSOMA_DEPS="$HOME/.unilab/isaacgym"
export UNILAB_BENCHMARK_HSGYM_PYTHON="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/bin/python3.8"
export UNILAB_BENCHMARK_HSGYM_LIB="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/miniconda3/envs/hsgym/lib"
```

## Validation

Validate the environment with the benchmark script. The benchmark loads robot
models from URDF, so you must provide your own URDF model tree
(`go1_description/`, `g1_description/`, ...) and point `--models-root` or
`UNILAB_BENCHMARK_MODELS_ROOT` at its root directory:

```bash
PYTHONPATH="$UNILAB_BENCHMARK_HOLOSOMA_DEPS/isaacgym/python" \
LD_LIBRARY_PATH="$UNILAB_BENCHMARK_HSGYM_LIB" \
uv run --no-project "$UNILAB_BENCHMARK_HSGYM_PYTHON" \
    scripts/benchmark/physics/benchmark_physics_step_isaacgym.py \
    --tasks g1_walk_flat --batch-sizes 256 --models-root "$UNILAB_BENCHMARK_MODELS_ROOT"
```

## Training and Evaluation

Once the external environment is installed, training works out of the box.
The worker runtime is discovered automatically from `~/.unilab/isaacgym`;
when using a custom install root, export `UNILAB_ISAACGYM_HOME` before
training. `g1_walk_flat` currently ships isaacgym owner configs for PPO and
SAC:

```bash
# SAC
uv run train --algo sac --task g1_walk_flat --sim isaacgym

# PPO
uv run train --algo ppo --task g1_walk_flat --sim isaacgym
```

Playback rendering is provided natively by IsaacGym: the interactive mode
opens the gym viewer inside the worker process, and the record mode renders
offscreen with a camera sensor and writes `play_video.mp4` (the camera tracks
env 0's root; the view is adjustable via `training.cam_distance` /
`cam_elevation` / `cam_azimuth`). `play_render_mode=auto` (the default)
selects the interactive viewer when a display is reachable
(`DISPLAY`/`WAYLAND_DISPLAY`) and falls back to recording on headless hosts;
recording requires a finite `training.play_steps` (the default configs
provide one).

```bash
# Training enters playback automatically (auto); headless servers record video
uv run train --algo sac --task g1_walk_flat --sim isaacgym

# Evaluate a trained checkpoint in the interactive viewer
uv run eval --algo sac --task g1_walk_flat --sim isaacgym \
    --render-mode interactive --load-run <run_dir_name>

# Record a video on headless hosts (force record)
uv run eval --algo sac --task g1_walk_flat --sim isaacgym \
    --render-mode record --load-run <run_dir_name> training.play_steps=800
```

Note: both the interactive viewer and camera capture require the worker sim
to run on a GPU (`env.isaacgym_device_id >= 0`); a CPU-pipeline sim has no
graphics context and render requests fail closed with an explanatory error.

Common overrides (Hydra arguments follow the command directly):

```bash
# Small smoke run: 64 environments, 3 iterations only
uv run train --algo sac --task g1_walk_flat --sim isaacgym \
    algo.num_envs=64 algo.max_iterations=3

# Pick the GPU used by the worker
uv run train --algo sac --task g1_walk_flat --sim isaacgym env.isaacgym_device_id=1
```

Cross-backend migration (sim2sim): the isaacgym owner configs are fully
contract-compatible with the mujoco owner under the audit guard
(`src/unilab/utils/sim2sim.py`, verdict TRANSFERABLE), so checkpoints of the
same task transfer across backends. Playback rendering works on isaacgym, so
cross-backend policy evaluation (playing a mujoco-trained checkpoint on
isaacgym, or vice versa) runs directly through the `uv run eval` commands
above.

## Manual Setup

If the automated script fails, the equivalent manual command sequence is:

```bash
export UNILAB_ISAACGYM_HOME="${UNILAB_ISAACGYM_HOME:-$HOME/.unilab/isaacgym}"
mkdir -p "$UNILAB_ISAACGYM_HOME"

# 1. Dedicated miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p "$UNILAB_ISAACGYM_HOME/miniconda3"
rm /tmp/miniconda.sh

# 2. Python 3.8 conda environment
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" install -y -n base -c conda-forge mamba
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/mamba" create -y -n hsgym python=3.8 -c conda-forge --override-channels

# 3. Ubuntu 24.04 GLIBCXX fix
"$UNILAB_ISAACGYM_HOME/miniconda3/bin/conda" install -y -n hsgym -c conda-forge libstdcxx-ng

# 4. Download (or reuse) the tarball and install IsaacGym
curl -fL --retry 3 "https://developer.nvidia.com/isaac-gym-preview-4" \
  -o "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz"
tar -xzf "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz" -C "$UNILAB_ISAACGYM_HOME"
"$UNILAB_ISAACGYM_HOME/miniconda3/envs/hsgym/bin/pip" install -e "$UNILAB_ISAACGYM_HOME/isaacgym/python"
```

## Troubleshooting

- **Tarball download or validation fails**: the script verifies the download
  is a valid gzip tarball. If it fails, delete
  `$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz` and re-run, or pass
  a manually downloaded package with `--tarball <path>`.
- **First INIT handshake times out (worker unresponsive)**: the first
  `gymtorch` import JIT-compiles a C++ extension (several minutes, cached
  under `~/.cache/torch_extensions/py38_cu121/gymtorch/`). The setup script's
  self-check pre-warms this compile. If a compile process was ever killed
  hard, a stale `lock` file in that directory blocks later loads forever —
  delete it and retry. The worker needs the env's `bin/` on `PATH` (for
  ninja); `IsaacGymBackend` injects it automatically.
- **`GLIBCXX_3.4.32 not found` on Ubuntu 24.04**: the prebuilt IsaacGym
  libraries link against a newer libstdc++ than the system provides. The setup
  script installs conda-forge `libstdcxx-ng` into the `hsgym` environment to
  fix this; at runtime, point `LD_LIBRARY_PATH` at that env's `lib/`.
- **`from isaacgym import gymapi` fails**: make sure `LD_LIBRARY_PATH` points
  at `$UNILAB_BENCHMARK_HSGYM_LIB` (the `lib/` directory of the hsgym env) and
  that `PYTHONPATH` includes `isaacgym/python`.
