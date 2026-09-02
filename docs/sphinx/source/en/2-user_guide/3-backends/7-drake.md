# Drake Backend

UniLab's Drake backend uses the experimental `drake-uni` Python distribution
(`import drake_uni`) and a native C++ batch extension linked against a local
Drake installation. Drake owns batched physics; UniLab continues to own task,
reward, observation, reset policy, and training orchestration.

The backend is registered and has owner YAMLs for selected PPO, APPO, and SAC
tasks. It is currently an experimental, Linux-first path. A registry entry or
owner YAML is not by itself evidence that a task has completed native training.

## Prerequisites

- Linux x86_64 is the only native platform currently documented as supported.
- Python `>=3.10,<3.14`, `uv`, a C++20 compiler, and Python development headers.
- A Drake C++ install prefix with:
  `include/drake/`, `include/pybind11/`, and `lib/libdrake.so`.
- Eigen and fmt development flags discoverable through `pkg-config` (or
  equivalent compiler flags in the local toolchain).
- A clean process with no `pydrake` module imported before Drake batch startup.

Drake C++ is an external toolchain. Installing the Python package does not
install Drake or build the native extension.

## Install and build

From a UniLab checkout, install the optional Python dependency:

```bash
make setup-drake
# Equivalent:
uv sync --extra mujoco --extra motrix --extra drake
```

For local DrakeUni development, use an editable install instead:

```bash
uv pip install -e /path/to/drake_uni
```

Build the extension with the same Python interpreter that will run UniLab.
The extension is written into the DrakeUni source tree and is not a portable
wheel artifact:

```bash
/path/to/unilab/.venv/bin/python \
  /path/to/drake_uni/scripts/build_drake_batch.py \
  --drake-home /path/to/drake/install
```

The generated file is `src/drake_uni/compiled/_drake_env_pool*`.
Rebuild it after changing Drake, Python, the compiler, or the active virtual
environment. Do not commit the generated extension.

## Assets

Robot meshes and textures may be downloaded from Hugging Face on the cold path
when a task is materialized. Pre-fetch the assets for a deterministic or
offline run:

```bash
uv run unilab-pull-assets --robot go1
uv run unilab-pull-assets --robot go2
```

## Training smoke

Select the backend through the canonical CLI. Do not use
`training.sim_backend=drake` as a standalone switch.

The small Stewart scene is a useful first native probe:

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 algo.num_envs=8 training.no_play=true
```

After assets are available, a locomotion probe can use an existing owner:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

These commands validate installation and the backend contract only; they do
not claim policy convergence.

## Playback and rendering

- `--render-mode none` runs evaluation without playback.
- `--render-mode record` keeps Drake as the physics owner and uses UniLab's
  MuJoCo playback helper as an offline visual renderer. MuJoCo and the visual
  assets are therefore still required for recording.
- Interactive Drake rendering is not implemented.

Example:

```bash
uv run eval --algo ppo --task stewart_balance --sim drake \
  --load-run -1 --render-mode none
```

## Unsupported boundaries

Drake batch currently fails closed for unsupported `pydrake`-mixed processes,
interactive rendering, reset domain-randomization payloads, and interval
perturbations that are not explicit body-force inputs. Keep these capabilities
disabled in Drake owner YAMLs until a native contract and test evidence exist.

## Evidence level

Support is evidence-graded: `Registered`, `Configured`, `Tested`,
`Benchmarked`, and `Recommended` are separate claims. Native tests and training
records must include the Drake version, compiler, Python ABI, task, algorithm,
environment count, thread count, and outcome before a support-matrix upgrade.
