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
:link: tooling/index
:link-type: doc
Export ONNX, inspect NaNs, send W&B logs, and export scenes.
:::

:::{grid-item-card} Manipulation Notes
:link: manipulation/index
:link-type: doc
Task-specific Allegro, Sharpa, and Go2+Airbot notes.
:::

::::

```{toctree}
:hidden:
:caption: User Guide

training/index
algorithms/index
backends/index
tasks/index
domain_randomization/index
terrain/index
tooling/index
manipulation/index
```
