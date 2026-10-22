# Installation

This page covers dependency setup only. Training commands and playback details
live in the getting-started and algorithm pages.

## Requirements

- Python `>=3.10,<3.14`, from `pyproject.toml`.
- `uv-wheelnext 0.9.14`, the pinned WheelNext-enabled uv build used for
  dependency sync and command execution. Stable uv releases do not yet select
  the current PyTorch wheel-variant metadata by default.
- Git, `curl`, `tar`, and a SHA-256 utility, used to clone the repository,
  install the pinned uv build, and fetch runtime assets.
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
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# Linux / macOS; on Windows use WSL or download the matching WheelNext archive.
scripts/tools/install_uv_wheelnext.sh .uv-wheelnext
export PATH="$PWD/.uv-wheelnext:$PATH"

# Recommended main-environment interpreter:
uv python install 3.13
```

UniLab accepts Python `3.10` through `3.13`; `3.13` is the recommended main
environment. External-worker backends are not part of the scoped Manager runtime
during issue #1811.

Choose one core setup path:

```bash
# Full default setup: MuJoCo backend, with shell completion.
make setup
```

`make setup` syncs the default environment (MuJoCo backend and the built-in
RL stack) and installs shell completion. If `make` is unavailable, run the
matching commands directly:

```bash
uv sync --extra mujoco
uv run --no-sync unilab-complete install
```

## Conda And Pip

The recommended path is the in-repo `make setup` (or `uv`) workflow. Conda
can serve as an outer environment for Python, CUDA, or system-library isolation,
but once the environment is active keep using the repository's `make` / `uv`
commands inside it:

```bash
conda create -n unilab python=3.13
conda activate unilab
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
scripts/tools/install_uv_wheelnext.sh .uv-wheelnext
export PATH="$PWD/.uv-wheelnext:$PATH"
make setup
```

The same WheelNext-enabled `uv` must remain first on `PATH` inside the Conda
environment.

From a source checkout, pip is a fallback path. Install the package first, then
add optional runtimes explicitly:

```bash
# Editable install for local development:
pip install -e .

# Regular install (omit -e) for a wheel-style deployment:
# pip install .

# MuJoCo (resolves the pinned mjbatch integration fork, built against
# mujoco==3.11.0):
pip install "mujoco~=3.11.0" "mjbatch @ git+https://github.com/unilabsim/mjbatch.git@cf4a83d"
```

The editable install points at the checkout; the regular install copies the
package and its task configs (`unilab/conf/`) into the environment. In both
cases, `train`, `eval`, and `demo` work from any directory, while logs and
checkpoints are written under the current working directory. Plain pip does not
select WheelNext accelerator variants, so use the pinned uv build for CUDA,
ROCm, and XPU environments. The `mjbatch` engine builds against the pinned
`mujoco==3.11.0`; for MJWarp and Genesis, prefer the uv paths above. Robot
meshes and textures are intentionally excluded from the wheel and downloaded on
the cold path from the `unilabsim/unilab-robots` dataset. Ensure the installed
package location is writable, or pre-fetch assets with
`uv run unilab-pull-assets` from a source checkout.

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

The tensor-only Manager runtime currently exposes five production backends
(`mujoco`, `mjwarp`, `genesis`, `newton`, and `motrix`) plus the scoped Drake
owner. Their simulator dependencies are optional.

```bash
# All scoped backends together.
uv sync --extra mujoco --extra mjwarp --extra genesis --extra newton --extra motrix
```

| Backend | Install path | Important prerequisites |
| --- | --- | --- |
| MuJoCo | `make setup` or `uv sync --extra mujoco` | Builds the pinned `mjbatch` fork (bound to `mujoco==3.11.0`) from source; a C++17 toolchain and Python development headers are required until prebuilt wheels exist (roadmap open item) |
| MJWarp | `uv sync --extra mujoco --extra mjwarp` | NVIDIA CUDA; on a single-GPU host the current CUDA device is used by default, while multi-GPU topology remains explicit |
| Genesis | `uv sync --extra genesis` | The validated path uses Linux x86_64, an NVIDIA GPU, and the pinned torch/Genesis versions |
| Newton | `uv sync --extra newton` | Linux CUDA with the pinned newton / MuJoCo-Warp / Warp versions |
| Motrix | `uv sync --extra motrix` | CPU-authoritative MotrixSim with packed Torch HOST_BRIDGE transfers |
| Drake | `make setup-drake` | Local Drake C++ prefix plus the DrakeUni native batch extension; currently scoped to PPO `go2_joystick_flat` |

The `isaacgym`, `isaacsim`, and `superdex` adapters remain temporarily shelved
by `unisim-core` during issue #1811. Their extras and historical backend pages
are not production support claims and the UniLab train/eval CLI rejects them
until new capability, parity, and support-matrix evidence is provided.

Read the scoped backend pages for runtime variables, renderer requirements, and
verification commands:

- {doc}`MuJoCo <../2-user_guide/3-backends/1-mujoco>`
- {doc}`MJWarp <../2-user_guide/3-backends/0-index>`
- {doc}`Genesis <../2-user_guide/3-backends/5-genesis>`
- {doc}`Newton <../2-user_guide/3-backends/7-newton>`
- {doc}`Motrix <../2-user_guide/3-backends/2-motrix>`
- {doc}`Drake <../2-user_guide/3-backends/6-drake>`

## Algorithm Extras

No algorithm extra is required. The full RL stack ships inside the `unilab`
distribution itself under `src/unilab/rl/`: PPO runs directly on
`rsl-rl-lib`, and the APPO, off-policy (SAC / FlashSAC), and multi-GPU
data-parallel runtimes are included in the base install. The former
`unilab-rl` package is merged into this repository (issue #2372, decision in
{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`).

The legacy `uni_rl` extra has been removed. Do not pass it to `uv sync` or
`pip install`; the RL stack is included with the base distribution.

If your code imports `uni_rl` directly, see
{doc}`../3-deployment/3-framework_migration/7-from_unilab_rl` for the
`uni_rl.*` → `unilab.rl.*` migration mapping.

## Switching The Local MuJoCo Version

The `mujoco` extra declares `mujoco~=3.11.0`, and the `mjbatch` batch engine
is built against `mujoco==3.11.0`: it records its build-time mujoco version
and refuses to import against a different one (see the watchdog row in
"Install Error Signatures"). Switching the local MuJoCo version therefore
requires an `mjbatch` build against that version — it is not a UniLab config
change:

1. bump the `mujoco` bound and the `mjbatch` source pin in `pyproject.toml`,
2. re-lock (`uv lock`) and re-sync (`uv sync --extra mujoco`).

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

## Accelerator Variants

UniLab uses one wheel-variant-aware `uv.lock`. During sync, the WheelNext
provider plugins inspect the host and select a CPU, CUDA, ROCm, or XPU Torch
wheel from the variants recorded in the lock. There are no accelerator profiles
to activate and no tracked `pyproject.toml` / `uv.lock` files to swap. See
{doc}`/adr/ADR-0020-wheel-variant-dependency-path`.

On Apple Silicon macOS, use `make setup` for the MuJoCo training path. MuJoCo
playback uses the `mjpython` application bundled by the official MuJoCo wheel. Torch's
`mps` device is selected automatically when available, and the portable
`cuda` alias resolves to MPS when CUDA is absent.

On Windows, use WSL or a matching WheelNext uv archive for dependency sync. The
`mjbatch` engine does not ship Windows wheels (the official `mujoco.dll` provides
no import library), so the MuJoCo physics backend stays Linux/macOS-only there;
plain `mujoco` (MJCF conversion, playback rendering via other backends) still
installs.

Accelerator notes:

- The pinned `uv-wheelnext 0.9.14` build is required. Verify it with
  `uv --version`; stable uv releases do not yet consume the current variant
  metadata correctly.
- The host still needs a compatible accelerator kernel driver. ROCm Torch wheels
  carry their userspace, so the host-side ROCm userspace version need not exactly
  match the wheel. The `rocm7.0` Torch variant has been validated on an 8-GPU
  MI300X host with a compatible `6.19` AMD kernel driver.
- Intel XPU hosts need their Level Zero driver stack. On Ubuntu 24.04+, install
  `intel-opencl-icd` and `libze-intel-gpu1`; off-policy training can add
  `training.use_amp=true` as needed.
- The training device field keeps Torch's `cuda` spelling on ROCm/HIP; do not
  set it to `rocm`.
- Run `uv run unilab doctor` after sync or whenever GPU behavior looks wrong. It
  detects AMD kernel/ROCm management interfaces, reports the installed Torch
  build, and fails when an AMD host has a CUDA or CPU Torch wheel.
- With ROCm PyTorch and an available GPU, a `HOST_BRIDGE` backend declaring and
  accepting `cuda` buffers uses the current GPU for Manager/TorchEnv tensors
  whenever the training learner resolves to CUDA: the default
  `training.collector_tensor_device=auto` follows the learner device, and an
  explicit `training.collector_tensor_device=cuda` forces the same routing for
  other learner devices. Direct environment construction remains CPU.
  Host-bridge physics stays on CPU; packed transfers connect it to the GPU
  observations, actions, rewards, and resets. CUDA-only physics backends remain
  unsupported on ROCm. See
  {doc}`/adr/ADR-0012-sole-tensor-manager-and-scoped-backends`.
- Downstream applications use the same WheelNext-enabled uv build. In a new
  project environment, `uv pip install unilab` selects the accelerator-aware
  Torch variant; plain pip remains a CPU/plain-wheel fallback.

## Package Mirrors And Variants

The current wheel-variant path must resolve the WheelNext variant metadata
index. Do not replace the default index with a plain PyPI mirror while syncing
this lock; mirror support should be revisited after stable uv/PyPI variant
metadata integration.

## Smoke Check

After sync, run a small check through the top-level CLI:

```bash
uv run unilab doctor
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

Do not use the `training.sim_backend` field by itself to switch backends; choose
the backend with `--sim`.
