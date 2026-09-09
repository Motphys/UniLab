# Installation

This page covers dependency setup only. Training commands and playback details
live in the getting-started and algorithm pages.

## Requirements

- Python `>=3.10,<3.14`, from `pyproject.toml`.
- `uv`, used for dependency sync and command execution.
- Git and `curl`, used to clone the repository and fetch runtime assets.
- `cmake`, required when building the Drake native batch extension. The Drake
  setup script uses CMake and a C++ toolchain.
- For the `mujoco` extra: the default install path uses the prebuilt
  `mujoco-uni-runtime` wheel (bound to `mujoco==3.11.0`), so
  `make setup` / `uv sync --extra mujoco` needs no compiler. A C++17 toolchain
  and Python development headers are only required on the explicit
  source-rebuild path (switching the MuJoCo version; see "Switching The Local
  MuJoCo Version"); without them the build fails with errors such as
  `fatal error: Python.h: No such file or directory` (see "Install Error
  Signatures" for the full lookup table).
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

# MuJoCo only.
# make setup-mujoco
```

`make setup` runs `uv sync --extra mujoco --extra motrix` and installs shell
completion. `make setup-motrix` runs `uv sync --extra motrix` and installs the
same completion entry. `make setup-mujoco` runs `uv sync --extra mujoco` and
installs completion. Run only one of these paths. If `make` is unavailable, run
the matching commands directly:

```bash
# Full default setup:
uv sync --extra mujoco --extra motrix
uv run --no-sync unilab-complete install

# Motrix only:
# uv sync --extra motrix && uv run --no-sync unilab-complete install

# MuJoCo only:
# uv sync --extra mujoco && uv run --no-sync unilab-complete install
```

## Conda And Pip

The recommended path is still the in-repo `make setup` / `make setup-motrix` (or
`uv`) workflow. Use `make setup-mujoco` when Motrix is not needed. Conda can
serve as an outer environment for Python, CUDA, or system-library isolation,
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

Use `make setup-mujoco` if you do not need Motrix. ROCm and XPU still go through
the platform-specific `make` targets below.

From a source checkout, pip is a fallback path. Install the package first, then
add optional runtimes explicitly:

```bash
# Editable install for local development:
pip install -e .

# Regular install (omit -e) for a wheel-style deployment:
# pip install .

# Motrix, when needed:
pip install motrixsim-core==0.8.2

# MuJoCo, when needed (the default install resolves the prebuilt wheel bound
# to mujoco==3.11.0):
pip install "mujoco~=3.11.0" "mujoco-uni-runtime==0.5.0"
```

The editable install points at the checkout; the regular install copies the
package and its task configs (`unilab/conf/`) into the environment. In both
cases, `train`, `eval`, and `demo` work from any directory, while logs and
checkpoints are written under the current working directory. The prebuilt
`mujoco-uni-runtime` wheel installs directly through pip with no build step;
`pybind11` / `wheel` and `--no-build-isolation` are only needed when forcing an
sdist rebuild against a non-default mujoco version (see "Switching The Local
MuJoCo Version"). For MJWarp, Genesis, platform-specific torch indexes, and
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
| MuJoCo | `make setup-mujoco` or `uv sync --extra mujoco` | Prebuilt wheel (bound to `mujoco==3.11.0`), no compiler needed; a C++17 toolchain and Python development headers are only required when switching versions (`make mujoco MJ=<version>`) |
| Motrix | `make setup-motrix` or `uv sync --extra motrix` | Motrix runtime is installed from the pinned Python package |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA and an explicit CUDA process device |
| Genesis | `uv sync --extra genesis` | The validated path uses Linux x86_64, an NVIDIA GPU, and the pinned torch/Genesis versions |
| Newton | `uv sync --extra newton` | NVIDIA CUDA; can be combined with the `mujoco` / `mjwarp` extras in one environment |
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
- {doc}`IsaacGym <../2-user_guide/3-backends/3-isaacgym>`
- {doc}`IsaacSim <../2-user_guide/3-backends/4-isaacsim>`

## Switching The Local MuJoCo Version

The default install path of the `mujoco` extra uses the prebuilt
`mujoco-uni-runtime==0.5.0` wheel. Each runtime release carries exactly one
prebuilt MuJoCo binding: the 0.5.0 wheels are compiled against
`mujoco==3.11.0`, and the native extension records its build-time mujoco
version and refuses to load on a mismatch (see the watchdog row in "Install
Error Signatures"). A bump of the default MuJoCo version therefore always
ships with a new runtime release; the coordination details live in the
mujoco-uni-runtime repository's `docs/release-coordination.md`.

Switching the MuJoCo version inside the support window `>=3.5,<3.12` always
takes the source-rebuild path:

```bash
make mujoco MJ=3.10.0
```

The `mujoco` extra declares `mujoco~=3.11.0`, which a re-lock can never leave,
so the target operates on the environment directly (`uv pip`, without touching
`uv.lock`). It runs, in order:

1. the `check-cxx-toolchain` preflight: fails fast when no C++ compiler is
   found and prints per-platform install commands;
2. `uv pip install "mujoco==3.10.0" pybind11 wheel setuptools`: installs the
   requested mujoco plus the runtime's build requirements into the current
   environment;
3. `uv cache clean mujoco-uni-runtime`: drops the build cache (uv's cache
   cannot see that the extension depends on the mujoco version);
4. `uv pip install --force-reinstall --no-deps --no-build-isolation
   --no-binary mujoco-uni-runtime "mujoco-uni-runtime==<installed version>"`:
   recompiles the native extension from the sdist against the new mujoco.

The override is **environment-local**: `uv.lock` stays unchanged. The
switch-back path is `uv sync --extra mujoco --reinstall-package
mujoco-uni-runtime`, which restores the locked default (mujoco 3.11.0 +
prebuilt wheel); the `--reinstall-package` flag is required because a plain
`uv sync` restores mujoco but keeps the locally rebuilt extension, which then
fails to load. The source rebuild requires a C++17 toolchain and Python
development headers (see "Requirements").

## Install Error Signatures

Reverse-lookup from error text to cause and fix.

| Error signature | Where it comes from | Fix |
| --- | --- | --- |
| `error: building mujoco-uni-runtime from source requires a C++ toolchain, but 'c++' was not found.` | The `check-cxx-toolchain` preflight of `make mujoco MJ=<version>` | Install a C++ toolchain and retry: Debian/Ubuntu `sudo apt-get install build-essential`; macOS `xcode-select --install`; Fedora/RHEL `sudo dnf install gcc-c++ make` |
| `error: [Errno 2] No such file or directory: 'c++'` (or `c++: No such file or directory`) | Building `mujoco-uni-runtime` from the sdist without a compiler; only occurs on the source-rebuild path — the default wheel path never compiles | Same toolchain install as above; or do not switch versions and use the default wheel path `uv sync --extra mujoco` |
| `fatal error: Python.h: No such file or directory` | Missing Python development headers during a source rebuild | A uv-managed Python (`uv python install`) bundles the headers; system Pythons need `python3-dev` (Debian/Ubuntu) or `python3-devel` (Fedora/RHEL) |
| `MuJoCoUni native batch extension was built against mujoco '3.11.0', but loaded mujoco is '...'` | Version watchdog: the extension's recorded build-time mujoco version does not match the loaded mujoco | Install the mujoco version the extension binds (the prebuilt wheel binds `3.11.0`: `uv sync --extra mujoco --reinstall-package mujoco-uni-runtime`); or rebuild from source against the active mujoco: `make mujoco MJ=<version>` |
| `mujoco_uni 0.5.0 supports official mujoco>=3.5,<3.12; found mujoco '...'` | The installed mujoco is outside the runtime's support window | Install a mujoco version inside `>=3.5,<3.12` (`make mujoco MJ=<version>`) |
| `MuJoCoUni native batch extension has not been built` | The native extension failed to import (`mujoco_uni.batch_available()` returns `False`); a common cause is a plain `uv sync` after a version switch, which restores mujoco but keeps the locally rebuilt extension linked to the old `libmujoco.so` | Run `uv run python -c "import mujoco_uni; print(mujoco_uni.batch_import_error())"` for the underlying cause; after a version switch, restore the prebuilt wheel with `uv sync --extra mujoco --reinstall-package mujoco-uni-runtime` |

## Platform Profiles

Linux CUDA and macOS use the default `pyproject.toml`. The default Linux torch
wheel source is the PyTorch `cu128` index configured in `pyproject.toml`.

On Apple Silicon macOS, `make setup-motrix` is the shortest interactive path.
The CLI routes Motrix playback through `mxpython` when needed; MuJoCo playback
uses the `mjpython` application bundled by the official MuJoCo wheel. Torch's
`mps` device is selected automatically when available, and the portable
`cuda` alias resolves to MPS when CUDA is absent.

On Windows, use the direct `uv sync` commands from above unless GNU `make` and
Bash are available. The default install uses the prebuilt wheel; MSVC Build
Tools and Python development headers are only needed when rebuilding the MuJoCo
native extension from source (version switch). If you want to use the Makefile,
install GNU Make and Bash separately (for example through Chocolatey or WSL).

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
