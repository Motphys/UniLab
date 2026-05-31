# Training

Training in UniLab is config-first. Use the package CLI for day-to-day runs and
the script entrypoints when debugging the underlying Hydra composition.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} CLI reference
:link: 1-cli_reference
:link-type: doc
Routes for `uv run train`, `uv run eval`, `uv run demo`, and low-level scripts.
:::

:::{grid-item-card} Hydra config
:link: 2-hydra_config
:link-type: doc
Owner YAML layout, backend selection, and safe override examples.
:::

:::{grid-item-card} Logs and tracking
:link: 3-logging
:link-type: doc
TensorBoard, W&B, run metadata, and trace options.
:::

:::{grid-item-card} Resume and checkpoints
:link: 5-resume_and_checkpoints
:link-type: doc
How `algo.load_run`, checkpoint files, and replay commands fit together.
:::

:::{grid-item-card} Docker
:link: 6-docker
:link-type: doc
Run UniLab inside the checked-in Linux NVIDIA image workflow.
:::

:::{grid-item-card} Multi-GPU
:link: 4-multi_gpu
:link-type: doc
Current off-policy multi-GPU knobs and their config boundary.
:::

::::

## When to Drop to `scripts/train_*.py`

Day-to-day runs should use the unified CLI. Reach for the low-level
`scripts/train_*.py` entrypoints only when you are:

- debugging a specific training stack,
- observing Hydra compose behavior directly, or
- comparing script-level log directories or adapter behavior.

## Related

- Pick an algorithm: {doc}`../2-algorithms/0-index`
- Find task commands: {doc}`../4-tasks/0-index`
- Compare backend behavior: {doc}`../3-backends/0-index`
- Check exact support status: {doc}`../../5-reference/5-support_matrix`

```{toctree}
:hidden:

1-cli_reference
2-hydra_config
3-logging
4-multi_gpu
5-resume_and_checkpoints
6-docker
```
