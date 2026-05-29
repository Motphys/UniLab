# Training

Training in UniLab is config-first. Use the package CLI for day-to-day runs and
the script entrypoints when debugging the underlying Hydra composition.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} CLI reference
:link: cli_reference
:link-type: doc
Routes for `uv run train`, `uv run eval`, `uv run demo`, and low-level scripts.
:::

:::{grid-item-card} Hydra config
:link: hydra_config
:link-type: doc
Owner YAML layout, backend selection, and safe override examples.
:::

:::{grid-item-card} Logs and tracking
:link: logging
:link-type: doc
TensorBoard, W&B, run metadata, and trace options.
:::

:::{grid-item-card} Resume and checkpoints
:link: resume_and_checkpoints
:link-type: doc
How `algo.load_run`, checkpoint files, and replay commands fit together.
:::

:::{grid-item-card} Docker
:link: docker
:link-type: doc
Run UniLab inside the checked-in Linux NVIDIA image workflow.
:::

:::{grid-item-card} Multi-GPU
:link: multi_gpu
:link-type: doc
Current off-policy multi-GPU knobs and their config boundary.
:::

::::

```{toctree}
:hidden:

cli_reference
hydra_config
logging
multi_gpu
resume_and_checkpoints
docker
```
