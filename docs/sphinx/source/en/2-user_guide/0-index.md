# User Guide

Daily usage reference once the repo is installed. If you are setting up UniLab
for the first time, start with {doc}`../1-getting_started/0-index`.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Training
:link: 1-training/0-index
:link-type: doc
CLI routes, Hydra owner YAMLs, logs, checkpoints, Docker, and multi-GPU notes.
:::

:::{grid-item-card} Algorithms
:link: 2-algorithms/0-index
:link-type: doc
Compare PPO, APPO, SAC, TD3, FlashSAC, MLX PPO, HIM-PPO, and HORA.
:::

:::{grid-item-card} Backends
:link: 3-backends/0-index
:link-type: doc
Choose MuJoCo or Motrix from owner YAMLs and backend capability evidence.
:::

:::{grid-item-card} Tasks
:link: 4-tasks/0-index
:link-type: doc
Find locomotion, motion tracking, manipulation, and mobile manipulation tasks.
:::

:::{grid-item-card} Domain Randomization
:link: 5-domain_randomization/0-index
:link-type: doc
Configure reset, init, and interval randomization through task owner configs.
:::

:::{grid-item-card} Tooling
:link: 7-tooling/0-index
:link-type: doc
Export ONNX, inspect NaNs, send W&B logs, and export scenes.
:::

:::{grid-item-card} Manipulation Notes
:link: 8-manipulation/0-index
:link-type: doc
Task-specific Allegro, Sharpa, and Go2+Airbot notes.
:::

::::

```{toctree}
:hidden:
:caption: User Guide

1-training/0-index
2-algorithms/0-index
3-backends/0-index
4-tasks/0-index
5-domain_randomization/0-index
6-terrain/0-index
7-tooling/0-index
8-manipulation/0-index
```
