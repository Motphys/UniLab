# User Guide

Daily usage reference once the repo is installed. If you are setting up UniLab
for the first time, start with {doc}`../getting_started/index`.

```{div} feature-list

- Pick a backend (**MuJoCo** for fidelity, **Motrix** for throughput).
- Pick an algorithm (PPO is the default; SAC variants for off-policy).
- Run on whatever GPU you have (CUDA / MPS / ROCm / XPU).
- Export: ONNX for deployment-oriented playback paths.

```

## Training Basics

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🧰 Install
:link: ../getting_started/installation
:link-type: doc
`uv` setup, GPU stacks, common pitfalls.
:::

:::{grid-item-card} ⚡ Quickstart
:link: ../getting_started/quickstart
:link-type: doc
First training run in three commands.
:::

:::{grid-item-card} 🎓 Training in depth
:link: ../getting_started/training
:link-type: doc
Runner lifecycle, checkpoints, distributed.
:::

:::{grid-item-card} 🧪 Config overrides
:link: ../getting_started/configuration_overrides
:link-type: doc
The Hydra `key=value` cheatsheet.
:::

:::{grid-item-card} 🎬 Evaluation & playback
:link: ../getting_started/evaluation_and_playback
:link-type: doc
Headless render, MP4 export, replay.
:::

::::

## Simulation Backends

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📦 Backend overview
:link: backends/index
:link-type: doc
The contract every backend must obey.
:::

:::{grid-item-card} 🧮 MuJoCo
:link: backends/mujoco
:link-type: doc
Reference backend; highest fidelity.
:::

:::{grid-item-card} 🚀 Motrix
:link: backends/motrix
:link-type: doc
High-throughput backend; default for locomotion training.
:::

:::{grid-item-card} 🤔 Choosing a backend
:link: backends/choosing_a_backend
:link-type: doc
When to switch — feature parity, perf, gotchas.
:::

::::

## Algorithms

```{toctree}
:maxdepth: 1

algorithms/overview
algorithms/ppo
algorithms/appo
algorithms/fast_sac
algorithms/fast_td3
algorithms/flash_sac
algorithms/him_ppo
algorithms/hora
algorithms/mlx_ppo
```

## Tasks

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🦿 G1 motion tracking
:link: tasks/g1_motion_tracking
:link-type: doc
Whole-body humanoid motion tracking.
:::

:::{grid-item-card} 🐕 Go2 + arm manip-loco
:link: tasks/go2_arm_manip_loco
:link-type: doc
Quadruped with a mounted arm.
:::

:::{grid-item-card} 🏃 Locomotion zoo
:link: tasks/locomotion_zoo
:link-type: doc
Joystick, rough terrain, stair, Go2W wheels.
:::

:::{grid-item-card} 🤚 Manipulation zoo
:link: tasks/manipulation_zoo
:link-type: doc
Allegro / Sharpa in-hand and grasp generation.
:::

::::

## Domain Randomization

```{toctree}
:maxdepth: 1

domain_randomization/index
domain_randomization/recipes
```

## Terrain

```{toctree}
:maxdepth: 1

terrain/procedural
terrain/heightfield_import
```

## Manipulation

```{toctree}
:maxdepth: 1

manipulation/dexterous_inhand
manipulation/manip_loco
```

## Tooling

```{toctree}
:maxdepth: 1

tooling/wandb_and_tensorboard
tooling/onnx_export
tooling/nan_visualizer
tooling/scene_export
```

```{toctree}
:hidden:
:caption: Training Basics

../getting_started/installation
../getting_started/quickstart
../getting_started/training
../getting_started/configuration_overrides
../getting_started/evaluation_and_playback
```

```{toctree}
:hidden:
:caption: Backends

backends/index
backends/mujoco
backends/motrix
backends/choosing_a_backend
```

```{toctree}
:hidden:
:caption: Tasks

tasks/g1_motion_tracking
tasks/go2_arm_manip_loco
tasks/locomotion_zoo
tasks/manipulation_zoo
```

## Looking for the Chinese version?

See {doc}`/zh_CN/user_guide/index`.
