# IsaacGym Backend

IsaacGym (NVIDIA Preview 4) is an end-of-life GPU physics simulator from NVIDIA
that only supports Python 3.6-3.8. The UniLab main environment requires
Python >= 3.10, so IsaacGym cannot be installed into it; it is used through an
external, standalone Python 3.8 environment located purely via environment
variables, with no machine-local paths written into the repository.

Current status: `IsaacGymBackend` (a subprocess backend whose physics runs in
the external Python 3.8 worker) is implemented and registered; `g1_walk_flat`
ships isaacgym owner configs
(`conf/{ppo,appo,sac,td3,flashsac}/task/g1_walk_flat/isaacgym.yaml`), and the
cross-backend contract audit (`scripts/audit_sim2sim_contracts.py`) covers the
mujoco/isaacgym pair. Playback rendering is not supported yet (owner configs
set `play_render_mode: none`), and the top-level CLI `--sim` flag does not
list isaacgym yet (same as drake; the backend is selected through the owner
YAML). Real-machine end-to-end validation (MJCF import fidelity, etc.)
depends on the external environment described below and is not covered by
repo CI. The repository also ships a physics benchmark script
`scripts/benchmark/physics/benchmark_physics_step_isaacgym.py`, which locates
the external environment through variables such as
`UNILAB_BENCHMARK_HOLOSOMA_DEPS`. This page covers preparing the external
environment, validating it with the benchmark, and the current integration
status.

## Prerequisites

- Linux x86_64 with an NVIDIA GPU driver installed.
- An NVIDIA developer account: `IsaacGym_Preview_4_Package.tar.gz` must be
  downloaded manually from <https://developer.nvidia.com/isaac-gym-preview-4>
  after logging in; the script cannot and will not fetch that URL for you.
- Disk space: roughly 5 GB for miniconda, the conda environment, and the
  IsaacGym package combined.

## Automated Setup

From the repository root, run:

```bash
scripts/tools/setup_isaacgym_env.sh
```

The script installs everything under `$HOME/.unilab/isaacgym` by default;
override the install root with the `UNILAB_ISAACGYM_HOME` environment variable,
and point at the package with `--tarball <path>` (default:
`$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz`). The script is
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

# 4. Unpack and install IsaacGym (tarball downloaded manually, see prerequisites)
tar -xzf "$UNILAB_ISAACGYM_HOME/IsaacGym_Preview_4_Package.tar.gz" -C "$UNILAB_ISAACGYM_HOME"
"$UNILAB_ISAACGYM_HOME/miniconda3/envs/hsgym/bin/pip" install -e "$UNILAB_ISAACGYM_HOME/isaacgym/python"
```

## Troubleshooting

- **Fetching the NVIDIA download URL with wget fails**: the download requires a
  login and must be done manually; see Prerequisites.
- **`GLIBCXX_3.4.32 not found` on Ubuntu 24.04**: the prebuilt IsaacGym
  libraries link against a newer libstdc++ than the system provides. The setup
  script installs conda-forge `libstdcxx-ng` into the `hsgym` environment to
  fix this; at runtime, point `LD_LIBRARY_PATH` at that env's `lib/`.
- **`from isaacgym import gymapi` fails**: make sure `LD_LIBRARY_PATH` points
  at `$UNILAB_BENCHMARK_HSGYM_LIB` (the `lib/` directory of the hsgym env) and
  that `PYTHONPATH` includes `isaacgym/python`.
