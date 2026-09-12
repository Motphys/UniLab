# Installation

This page covers dependency setup only. Training commands and playback details
live in the getting-started and algorithm pages.

## Requirements

- Python `>=3.10,<3.14`, from `pyproject.toml`.
- `uv`, used for dependency sync and command execution.
- Git and `curl`, used to clone the repository and fetch runtime assets.
- `cmake`, required when building the Drake native batch extension. The Drake
  setup script uses CMake and a C++ toolchain.
- For the `mujoco` extra: the MuJoCo physics backend executes on the
  `mjbatch` native batch engine, currently consumed from the pinned
  integration fork (`unilabsim/mjbatch`). uv builds it from the pinned git
  source with an isolated build (scikit-build-core + nanobind), which requires
  a C++17 toolchain and Python development headers; the build binds
  `mujoco==3.11.0` and the engine refuses to import against any other mujoco
  version. A no-compiler install path depends on the fork's final distribution
  channel (prebuilt wheels vs the git pin), which is the roadmap's open item
  (see "Switching The Local MuJoCo Version" and "Install Error Signatures").
  - macOS: `xcode-select --install`
  - Ubuntu / Debian: `sudo apt-get install build-essential python3-dev`
  - Fedora / RHEL: `sudo dnf install gcc-c++ make python3-devel`
  - Windows: MSVC Build Tools
  - Tip: a uv-managed Python (`uv python install`) already bundles the
    headers, so `python3-dev` is only needed for system Pythons.

## Clone And Sync

```bash
# Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/unilabsim/UniLab.git
cd UniLab
# Recommended main-environment interpreter:
uv python install 3.13
```

UniLab accepts Python `3.10` through `3.13`; `3.13` is the recommended main
environment. IsaacGym and IsaacSim use their own worker Python versions below.

If you plan to use Drake, install CMake on the host as well:

```bash
# macOS:
brew install cmake

# Ubuntu / Debian:
# sudo apt-get install cmake
```

Choose one core setup path:

```bash
# Full default setup: MuJoCo + Motrix, with shell completion.
make setup

# Fastest path for the first Motrix demo.
# make setup-motrix
```

`make setup` runs `uv sync --extra mujoco --extra motrix` and installs shell
completion. `make setup-motrix` runs `uv sync --extra motrix` and installs the
same completion entry. Run only one of these paths. If `make` is unavailable,
run the matching commands directly:

```bash
# Full default setup:
uv sync --extra mujoco --extra motrix
uv run --no-sync unilab-complete install

# Motrix only:
# uv sync --extra motrix && uv run --no-sync unilab-complete install
```

## Conda And Pip

The recommended path is still the in-repo `make setup` / `make setup-motrix` (or
`uv`) workflow. Use `uv sync --extra mujoco` when Motrix is not needed. Conda
can serve as an outer environment for Python, CUDA, or system-library isolation,
but once the environment is active keep using the repository's `make` / `uv`
commands inside it:

```bash
conda create -n unilab python=3.13
conda activate unilab
pip install uv
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
make setup-motrix
```

Use `uv sync --extra mujoco` if you do not need Motrix. ROCm and XPU still go
through the platform-specific `make` targets below.

From a source checkout, pip is a fallback path. Install the package first, then
add optional runtimes explicitly:

```bash
# Editable install for local development:
pip install -e .

# Regular install (omit -e) for a wheel-style deployment:
# pip install .

# Motrix, when needed:
pip install motrixsim-core==0.8.2

# MuJoCo, when needed (resolves the pinned mjbatch integration fork, built
# against mujoco==3.11.0):
pip install "mujoco~=3.11.0" "mjbatch @ git+https://github.com/unilabsim/mjbatch.git@cf4a83d"
```

The editable install points at the checkout; the regular install copies the
package and its task configs (`unilab/conf/`) into the environment. In both
cases, `train`, `eval`, and `demo` work from any directory, while logs and
checkpoints are written under the current working directory. The `mjbatch`
engine builds against the pinned `mujoco==3.11.0`; for MJWarp, Genesis,
platform-specific torch indexes, and
ROCm/XPU profiles, prefer the uv paths above. Robot meshes and
textures are intentionally excluded from the wheel and downloaded on the cold
path from the `unilabsim/unilab-robots` dataset. Ensure the installed package
location is writable, or pre-fetch assets with `uv run unilab-pull-assets` from a
source checkout. The isaacgym / isaacsim backends still assume a source
checkout; use their dedicated setup pages below.

## Runtime Assets

Large assets are not bundled into the wheel; they are downloaded lazily on
cold paths (first use of the owning feature) from Hugging Face dataset repos:

- [Robot meshes and textures](https://huggingface.co/datasets/unilabsim/unilab-robots)
- [Motion clips](https://huggingface.co/datasets/unilabsim/unilab-motions)
- [Scenes](https://huggingface.co/datasets/unilabsim/unilab-scenes)
- [Grasp caches](https://huggingface.co/datasets/unilabsim/unilab-caches)
- [Demo checkpoints](https://huggingface.co/datasets/unilabsim/unilab-checkpoints)

Pre-fetch robot assets with `uv run unilab-pull-assets`. For mainland China,
set `HF_ENDPOINT=https://hf-mirror.com` when the default Hugging Face endpoint
is unreachable.

## Backend Extras

The base package installs the public `unisim-core` contract; simulator-specific
dependencies are optional. Choose the path that matches your backend. The
commands below are alternatives for a single-backend environment. If you plan
to compare several in-process backends, combine their extras in one `uv sync`
command; external worker scripts remain separate.

For example, a local comparison environment can install MuJoCo, Motrix, MJWarp,
and Genesis together:

```bash
uv sync --extra mujoco --extra motrix --extra mjwarp --extra genesis
```

The `mujoco`, `mjwarp`, and `newton` extras share one MuJoCo 3.11 /
MuJoCo-Warp 3.11 / Warp 1.16 line and are jointly installable in a single
environment. The `newton` extra also includes Newton's native ViewerGL
rendering dependencies (offline record + interactive):

```bash
uv sync --extra mujoco --extra mjwarp --extra newton
```

| Backend | Install path | Important prerequisites |
| --- | --- | --- |
| MuJoCo | `make setup` or `uv sync --extra mujoco` | Builds the pinned `mjbatch` fork (bound to `mujoco==3.11.0`) from source; a C++17 toolchain and Python development headers are required until prebuilt wheels exist (roadmap open item) |
| Motrix | `make setup-motrix` or `uv sync --extra motrix` | Motrix runtime is installed from the pinned Python package |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA and an explicit CUDA process device |
| Genesis | `uv sync --extra genesis` | The validated path uses Linux x86_64, an NVIDIA GPU, and the pinned torch/Genesis versions |
| Newton | `uv sync --extra newton` | NVIDIA CUDA; can be combined with the `mujoco` / `mjwarp` extras in one environment |
| SuperDex | `uv sync --extra superdex` | Published wheels support Linux x86_64 with CPython 3.12/3.13 only; FR3 assets download from Hugging Face on first use (`SUPERDEX_ASSETS_PATH` overrides with a local checkout) |
| Drake | `make setup-drake` | C++20, Eigen/fmt/spdlog, and an existing Drake prefix or the script's download path |
| IsaacGym | `bash scripts/tools/setup_isaacgym_env.sh` | Linux x86_64, NVIDIA driver, and a separate Python 3.8 worker environment |
| IsaacSim | `bash scripts/tools/setup_isaacsim_env.sh` | Linux x86_64, NVIDIA CUDA, a separate Python 3.11 worker, and Kit EULA acceptance |

The Drake, IsaacGym, and IsaacSim setup scripts install their external runtime
outside the repository and can be re-run safely. They do not install the
external simulator into the main UniLab environment. Read the backend pages for
runtime variables, renderer requirements, and verification commands:

- {doc}`MuJoCo <../2-user_guide/3-backends/1-mujoco>`
- {doc}`Motrix <../2-user_guide/3-backends/2-motrix>`
- {doc}`MJWarp <../2-user_guide/3-backends/0-index>`
- {doc}`Genesis <../2-user_guide/3-backends/5-genesis>`
- {doc}`Drake <../2-user_guide/3-backends/6-drake>`
- {doc}`Newton <../2-user_guide/3-backends/7-newton>`
- {doc}`SuperDex <../2-user_guide/3-backends/8-superdex>`
- {doc}`IsaacGym <../2-user_guide/3-backends/3-isaacgym>`
- {doc}`IsaacSim <../2-user_guide/3-backends/4-isaacsim>`

## Switching The Local MuJoCo Version

The `mujoco` extra declares `mujoco~=3.11.0`, and the `mjbatch` batch engine
is built against `mujoco==3.11.0`: it records its build-time mujoco version
and refuses to import against a different one (see the watchdog row in
"Install Error Signatures"). Switching the local MuJoCo version therefore
requires an `mjbatch` build against that version — it is not a UniLab config
change:

1. bump the `mujoco` bound and the `mjbatch` source pin in `pyproject.toml`
   (and mirror `pyproject.rocm.toml`),
2. re-lock (`uv lock`, plus the ROCm lockfile via `make sync-rocm`) and
   re-sync (`uv sync --extra mujoco`).

The fork's build pins `mujoco==3.11.0` at build time, so the isolated build
always compiles against the matching mujoco. Until the fork's distribution
identity is decided (PyPI package vs git pin — the roadmap's open item),
coordinate version bumps with the
[fork](https://github.com/unilabsim/mjbatch).

## Install Error Signatures

Reverse-lookup from error text to cause and fix.

| Error signature | Where it comes from | Fix |
| --- | --- | --- |
| `fatal error: Python.h: No such file or directory` | Missing Python development headers while uv builds the `mjbatch` fork from source | A uv-managed Python (`uv python install`) bundles the headers; system Pythons need `python3-dev` (Debian/Ubuntu) or `python3-devel` (Fedora/RHEL) |
| `error: [Errno 2] No such file or directory: 'c++'` (or `c++: No such file or directory`) | Building `mjbatch` from the pinned git source without a compiler | Install a C++ toolchain (see "Requirements") and retry `uv sync --extra mujoco` |
| `mjbatch was built against MuJoCo <version> but <other> is installed` | Version watchdog: the engine's build-time mujoco pin does not match the installed mujoco | Restore the locked pair with `uv sync --extra mujoco`; a different mujoco version requires an `mjbatch` rebuild (see "Switching The Local MuJoCo Version") |

## Platform Profiles

Linux CUDA and macOS use the default `pyproject.toml`. The default Linux torch
wheel source is the PyTorch `cu128` index configured in `pyproject.toml`.

On Apple Silicon macOS, `make setup-motrix` is the shortest interactive path.
The CLI routes Motrix playback through `mxpython` when needed; MuJoCo playback
uses the `mjpython` application bundled by the official MuJoCo wheel. Torch's
`mps` device is selected automatically when available, and the portable
`cuda` alias resolves to MPS when CUDA is absent.

On Windows, use the direct `uv sync` commands from above unless GNU `make` and
Bash are available. The `mjbatch` engine does not ship Windows wheels (the
official `mujoco.dll` provides no import library), so the MuJoCo physics
backend stays Linux/macOS-only there; plain `mujoco` (MJCF conversion,
playback rendering via other backends) still installs. If you want to use the
Makefile, install GNU Make and Bash separately (for example through Chocolatey
or WSL).

ROCm and Intel XPU have explicit Makefile targets:

```bash
make sync-rocm
make sync-xpu
```

`make sync-rocm` copies `pyproject.rocm.toml` into `pyproject.toml` and syncs the
ROCm profile. `make sync-xpu` syncs Motrix dependencies without installing the
default torch package, then installs the XPU torch wheel through `uv pip`.

ROCm notes:

- `make sync-rocm` requires ROCm `>= 7.1` and installs the matching PyTorch wheel
  from the repository's ROCm dependency files.
- It swaps `pyproject.rocm.toml` / `uv.rocm.lock` in as the active
  `pyproject.toml` / `uv.lock`, so afterwards you can run bare `uv run ...`.
- To return to the default CUDA / macOS profile, run
  `git restore -- pyproject.toml uv.lock` and then re-run `make setup-motrix`
  (or `uv sync --extra motrix`); confirm the active profile before committing any
  non-ROCm dependency change.
- The training device field keeps `cuda` semantics; do not set it to `rocm`.
- When installing from PyPI instead of a source checkout, `make sync-rocm` does
  not apply. Install the torch build validated by the repository from the
  PyTorch ROCm index first, then `unilab`. The published dependency range is
  `torch>=2.8,<2.12`, so pip keeps the installed ROCm build instead of
  replacing it with the CUDA wheel:

  ```bash
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
  pip install unilab
  ```

Intel XPU notes:

- Keep using `uv run --no-sync ...` so the default Linux dependencies are not
  synced back in.
- Ubuntu 24.04+ also needs the system driver packages `intel-opencl-icd` and
  `libze-intel-gpu1`.
- Off-policy training can add `training.use_amp=true` as needed.

## Package Mirrors

For a local package mirror, set the uv index before syncing:

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra mujoco --extra motrix \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Smoke Check

After sync, run a small check through the top-level CLI:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

For Motrix, install the extra first and switch with `--sim`:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

Do not use the `training.sim_backend` field by itself to switch backends; choose
the backend with `--sim`.
