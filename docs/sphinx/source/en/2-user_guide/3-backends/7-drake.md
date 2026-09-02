# Drake Backend

This page is the end-to-end path for running UniLab with the Drake batch
backend. Follow the sections in order on a new machine: provision the native
toolchain, install the Python extra, build the extension, run diagnostics, then
prepare assets and launch a small training smoke.

Drake owns batched physics. UniLab still owns the task, reward, observation,
reset policy, and training orchestration. Rendering is provided by MuJoCo's
native renderer: Drake advances the state, and MuJoCo consumes that state only
to draw frames (it never advances a second physics simulation). The backend is
experimental and Linux-first; a registry entry or owner YAML is not, by itself,
evidence that a task has completed native training.

Use Drake when you need CPU-oriented batched physics and can provision the
external C++ toolchain. MuJoCo or Motrix may still offer broader task coverage,
but Drake playback uses MuJoCo for both recorded and interactive visualization.
Before selecting `--sim drake`, check that the algorithm/task has a Drake owner
YAML and review its evidence in the {doc}`../../5-reference/5-support_matrix`.

## System requirements

The complete native path is currently documented for Linux x86_64. The
official `noble` tarball is built for Ubuntu 24.04; that Ubuntu version (or a
distribution with compatible glibc/libstdc++ runtime libraries) is the tested
path. Other operating systems and architectures are not covered by the setup
script.

| Component | Requirement |
| --- | --- |
| Python | `>=3.10,<3.14` (the validated Drake run used Python 3.12; Python 3.12 or 3.13 is recommended) |
| Python environment | `uv` and a virtual environment created by the repository |
| Compiler | A C++20 compiler available as `c++` (or `CXX`) |
| Build/runtime tools | `git`, `curl`, `tar`, and `pkg-config` (the download path uses all four) |
| Headers/libraries | Eigen 3, fmt, and spdlog development packages; Python headers for the interpreter used to build the extension |
| Drake prefix | `include/drake/`, `include/pybind11/`, and `lib/libdrake.so` |

On Ubuntu or Debian, install the system packages before running the setup
script:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libeigen3-dev \
  libfmt-dev \
  libspdlog-dev \
  curl \
  git
```

If UniLab uses a system Python, also install its matching development headers
(for example `python3-dev`). A uv-managed Python normally includes the headers
needed by the extension build. `cmake` is not required by
`build_drake_batch.py`; the extension is compiled directly with the selected
C++ compiler.

## Recommended complete setup

Run this from the UniLab2 checkout:

```bash
bash scripts/tools/setup_drake_env.sh --download-drake
```

The script is resumable and idempotent. Downloads, completion markers, and the
combined log are kept under `~/.unilab/drake` by default. Useful options are:

```text
--drake-home <path>        use an existing Drake C++ prefix
--deps-root <path>         use an unpacked apt-style root for Eigen/fmt/spdlog
--drake-uni-source <path>  use a local DrakeUni checkout
--drake-version <version>  select the downloaded Drake version
--drake-platform <name>    select the official tarball platform (default: noble)
```

The script never installs system packages with `sudo`. It checks the prefix
before compiling and uses the UniLab virtual environment's Python, so the
generated extension matches the ABI that runs training.

### Environment variables

The setup script prints exports when it finishes, but a script cannot modify the
parent shell. In a new shell (or after sourcing these values), set:

```bash
export DRAKE_HOME="$HOME/.unilab/drake/drake-1.56.0-noble"
export UNILAB_DRAKE_HOME="$DRAKE_HOME"
export UNILAB_DRAKE_UNI_SOURCE="$PWD/../drake_uni"
export LD_LIBRARY_PATH="$DRAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

When `--deps-root /path/to/deps` was used, put its libraries before or
alongside Drake's libraries:

```bash
export LD_LIBRARY_PATH="/path/to/deps/usr/lib/x86_64-linux-gnu:$DRAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If DrakeUni is not the checkout next to UniLab2, replace the source path with
the checkout passed to `--drake-uni-source`. The generated file is
`src/drake_uni/compiled/_drake_env_pool*`; it is machine- and Python-ABI
specific and must not be committed. Rebuild it after changing Drake, Python,
the compiler, or the virtual environment.

## Existing Drake prefix or manual build

An existing prefix can be used without downloading:

```bash
bash scripts/tools/setup_drake_env.sh \
  --drake-home /opt/drake \
  --drake-uni-source /home/user/ws/unilabsim/drake_uni
```

The prefix must contain all of the following:

```text
/opt/drake/include/drake/
/opt/drake/include/pybind11/
/opt/drake/lib/libdrake.so
```

For a manual rebuild, use the same interpreter that runs UniLab:

```bash
uv pip install -e /home/user/ws/unilabsim/drake_uni
uv run --no-sync python \
  /home/user/ws/unilabsim/drake_uni/scripts/build_drake_batch.py \
  --drake-home /opt/drake
```

The build script discovers Eigen/fmt through `pkg-config`. If those packages
are outside standard system paths, set `EIGEN3_INCLUDE_DIR`, `FMT_INCLUDE_DIR`,
`FMT_LIB_DIR`, and `PKG_CONFIG_PATH` (or use the setup script's `--deps-root`).

## Verify the installation

Run the diagnostic in a clean process with Drake's libraries visible:

```bash
uv run --no-sync python - <<'PY'
import drake_uni
from drake_uni.runtime import batch_diagnostics

diagnostics = batch_diagnostics()
print("drake_uni:", drake_uni.__file__)
print("batch diagnostics:", diagnostics)
if not diagnostics.batch_available:
    raise SystemExit(diagnostics.batch_import_error or "Drake batch extension is unavailable")
PY
```

The successful result contains `batch_available=True`. If the extension is not
available, rerun the complete setup script and inspect its log under
`~/.unilab/drake/install.log`.

The focused contract and training tests are:

```bash
uv run --no-sync pytest \
  tests/base/backend/test_drake_batch_pool.py -q
uv run --no-sync pytest \
  tests/scripts/test_drake_training_smoke.py -m slow -q
```

The slow smoke intentionally runs one PPO iteration with four environments and
four steps per environment. It checks the real training entry point; it does
not measure convergence.

## Prepare robot assets

Robot meshes and textures are hosted on Hugging Face and are fetched on the
cold materialization path. Pre-fetch them for deterministic or offline runs:

```bash
uv run --no-sync unilab-pull-assets --robot go1
uv run --no-sync unilab-pull-assets --robot go2
# CI/offline image (downloads every registered robot):
uv run --no-sync unilab-pull-assets --robot all
```

The Stewart balance scene does not need the Go1/Go2 meshes, so it is the best
first probe. Go1, Go2, and Go2-arm tasks require their corresponding assets.
UniLab places files under its managed asset directories; do not move meshes by
hand.

## Run a training smoke

Select Drake with the top-level `--sim` flag. Do not use
`training.sim_backend=drake` as a standalone backend switch; that field is set
by the selected owner YAML.

Start with the compact Stewart scene:

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1
```

After pulling assets, check a locomotion owner with the same bounded settings:

```bash
uv run train --algo ppo --task go1_joystick_flat --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=4 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1

uv run train --algo ppo --task go2_joystick_flat --sim drake \
  algo.max_iterations=1 \
  algo.num_envs=4 \
  algo.num_steps_per_env=4 \
  training.no_play=true \
  env.drake_nthread=1
```

These are installation and contract probes, not recommended production
hyperparameters. For a real run, remove the one-iteration overrides and choose
an environment count appropriate for the host's CPU and memory:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake
```

Evaluate the latest run without opening a playback window:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim drake \
  --load-run -1 --render-mode none
```

With the default `training.play_render_mode=auto`, Drake playback records a
MuJoCo-rendered video automatically. No renderer-specific override is needed:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim drake --load-run -1
```

For an interactive MuJoCo window backed by Drake physics, use the shared viewer
entrypoint (the viewer is MuJoCo; `--sim drake` selects the physics owner):

```bash
uv run python -m unilab.scripts.play_interactive \
  --algo ppo --task go2_joystick_flat --sim drake \
  interactive.action_mode=policy
```

## Playback and rendering limits

- `--render-mode none` disables playback and is the safest mode for headless
  training/evaluation.
- Drake has no separate renderer. Both automatic recording and the interactive
  viewer use MuJoCo's native rendering APIs, while Drake remains the only
  physics engine being stepped.
- `--render-mode record` is an explicit spelling of the same automatic recording
  path; it still requires the MuJoCo extra and visual assets.
- `--render-mode none` is the only opt-out when rendering is not wanted.
- This is not sim-to-sim: MuJoCo is not stepped and the checkpoint is not
  re-evaluated under MuJoCo physics. The renderer receives the current Drake
  state solely for visualization.

## Troubleshooting

| Symptom | Check/fix |
| --- | --- |
| `ModuleNotFoundError: drake_uni` | Rerun `bash scripts/tools/setup_drake_env.sh --download-drake` in UniLab2. For an existing prefix, pass `--drake-home`; for a local checkout, pass `--drake-uni-source`. |
| `DrakeEnvPool batch extension has not been built` | Run the build script with the same `uv` Python used by UniLab; do not copy an extension built for another Python ABI. |
| `libdrake.so: cannot open shared object file` | Export `DRAKE_HOME` and prepend `$DRAKE_HOME/lib` (plus the `--deps-root` library directory, if applicable) to `LD_LIBRARY_PATH`. |
| Missing `include/drake` or `include/pybind11` | Point `--drake-home` at the Drake installation prefix, not at its parent download directory. |
| `fatal error: Python.h` | Install the matching `python3-dev` package, or recreate the environment with a uv-managed Python. |
| Eigen/fmt/spdlog headers or libraries not found | Install `libeigen3-dev`, `libfmt-dev`, and `libspdlog-dev`, or pass an unpacked dependency tree with `--deps-root`. |
| `robot mesh not found` | Pre-fetch the matching robot with `uv run --no-sync unilab-pull-assets --robot <robot>`. |
| Diagnostic reports `pydrake` already loaded | Start a fresh process and construct the Drake batch backend before importing `pydrake`; mixed processes fail closed. |

## Evidence level

Support claims are evidence-graded: `Registered`, `Configured`, `Tested`,
`Benchmarked`, and `Recommended` are separate claims. A native test or training
record should include the Drake version, compiler, Python ABI, task, algorithm,
environment count, thread count, and outcome before the support matrix is
upgraded.
