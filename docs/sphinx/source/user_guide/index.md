# User Guide

From a fresh checkout to a trained policy you can play back, headlessly
record, or deploy on real hardware.

```{div} feature-list

- Pick a backend (**MuJoCo** for fidelity, **Motrix** for throughput).
- Pick an algorithm (PPO is the default; SAC variants for off-policy).
- Run on whatever GPU you have (CUDA / MPS / ROCm / XPU).
- Ship: ONNX, Core ML, or just `.pt`.

```

## 1. Getting Started

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🧰 Install
:link: getting_started/installation
:link-type: doc
`uv` setup, GPU stacks, common pitfalls.
:::

:::{grid-item-card} ⚡ Quickstart
:link: getting_started/quickstart
:link-type: doc
First training run in three commands.
:::

:::{grid-item-card} 🎓 Training in depth
:link: getting_started/training
:link-type: doc
Runner lifecycle, checkpoints, distributed.
:::

:::{grid-item-card} 🧪 Config overrides
:link: getting_started/configuration_overrides
:link-type: doc
The Hydra `key=value` cheatsheet.
:::

:::{grid-item-card} 🎬 Evaluation & playback
:link: getting_started/evaluation_and_playback
:link-type: doc
Headless render, MP4 export, replay.
:::

::::

## 2. Simulation Backends

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

## 3. Algorithms

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

## 4. Tasks

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

## 5. Domain Randomization

```{toctree}
:maxdepth: 1

domain_randomization/index
domain_randomization/recipes
```

## 6. Terrain

```{toctree}
:maxdepth: 1

terrain/procedural
terrain/heightfield_import
```

## 7. Manipulation

```{toctree}
:maxdepth: 1

manipulation/dexterous_inhand
manipulation/manip_loco
```

## 8. Tooling

```{toctree}
:maxdepth: 1

tooling/wandb_and_tensorboard
tooling/onnx_export
tooling/nan_visualizer
tooling/scene_export
```

```{toctree}
:hidden:
:caption: Getting Started

getting_started/installation
getting_started/quickstart
getting_started/training
getting_started/configuration_overrides
getting_started/evaluation_and_playback
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

## 9. 中文文档(zh_CN)

```{toctree}
:maxdepth: 2
:caption: 中文用户指南

zh_CN/01-getting-started
zh_CN/02-simulation-backends
zh_CN/03-training
zh_CN/04-algorithms
zh_CN/05-domain-randomization
zh_CN/A-getting-started/01-install
zh_CN/A-getting-started/02-first-run
zh_CN/A-getting-started/03-docker
zh_CN/B-training/01-unified-cli
zh_CN/B-training/02-playback-and-resume
zh_CN/B-training/03-hydra-overrides
zh_CN/B-training/04-logging-and-wandb
zh_CN/B-training/05-docker
zh_CN/C-algorithms/01-ppo-torch
zh_CN/C-algorithms/02-mlx-ppo
zh_CN/C-algorithms/03-appo
zh_CN/C-algorithms/04-sac
zh_CN/C-algorithms/05-td3
zh_CN/C-algorithms/06-flashsac
zh_CN/D-tasks/01-task-index
zh_CN/D-tasks/02-g1-motion-tracking
zh_CN/D-tasks/03-allegro-inhand
zh_CN/D-tasks/04-sharpa-inhand
zh_CN/D-tasks/05-go2-rough-terrain
zh_CN/D-tasks/06-go2-arm-manip-loco
zh_CN/E-reference/01-backend-support-matrix
```
