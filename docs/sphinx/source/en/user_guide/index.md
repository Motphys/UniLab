# User Guide

Daily usage reference once the repo is installed. If you are setting up UniLab
for the first time, start with {doc}`../getting_started/index`.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Training
:link: training/index
:link-type: doc
CLI routes, Hydra owner YAMLs, logs, checkpoints, Docker, and multi-GPU notes.
:::

:::{grid-item-card} Algorithms
:link: algorithms/index
:link-type: doc
Compare PPO, APPO, SAC, TD3, FlashSAC, MLX PPO, HIM-PPO, and HORA.
:::

:::{grid-item-card} Backends
:link: backends/index
:link-type: doc
Choose MuJoCo or Motrix from owner YAMLs and backend capability evidence.
:::

:::{grid-item-card} Tasks
:link: tasks/index
:link-type: doc
Find locomotion, motion tracking, manipulation, and mobile manipulation tasks.
:::

:::{grid-item-card} Domain Randomization
:link: domain_randomization/index
:link-type: doc
Configure reset, init, and interval randomization through task owner configs.
:::

:::{grid-item-card} Tooling
:link: tooling/onnx_export
:link-type: doc
Export ONNX, inspect NaNs, send W&B logs, and export scenes.
:::

::::

```{toctree}
:hidden:
:caption: Training

training/index
training/cli_reference
training/hydra_config
training/logging
training/multi_gpu
training/resume_and_checkpoints
training/docker
```

```{toctree}
:hidden:
:caption: Algorithms

algorithms/index
algorithms/ppo
algorithms/appo
algorithms/sac
algorithms/td3
algorithms/flash_sac
algorithms/him_ppo
algorithms/hora
algorithms/mlx_ppo
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

tasks/index
tasks/locomotion
tasks/motion_tracking
tasks/manipulation
tasks/manip_loco
```

```{toctree}
:hidden:
:caption: Domain Randomization

domain_randomization/index
domain_randomization/configuration
domain_randomization/writing_providers
```

```{toctree}
:hidden:
:caption: Terrain

terrain/index
terrain/procedural
terrain/heightfield_import
```

```{toctree}
:hidden:
:caption: Tooling

tooling/onnx_export
tooling/wandb
tooling/nan_visualizer
tooling/scene_export
```

```{toctree}
:hidden:
:caption: Manipulation Notes

manipulation/dexterous_inhand
manipulation/manip_loco
```
