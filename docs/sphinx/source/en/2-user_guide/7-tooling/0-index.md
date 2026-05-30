# Tooling

Operational tools for exporting policies, inspecting training failures, sending
run metadata, and materializing scenes.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} ONNX export
:link: 1-onnx_export
:link-type: doc
Export policies from playback paths and verify runtime inputs.
:::

:::{grid-item-card} W&B and TensorBoard
:link: 2-wandb
:link-type: doc
Configure run logging and experiment metadata.
:::

:::{grid-item-card} NaN visualizer
:link: 3-nan_visualizer
:link-type: doc
Inspect NaN guard dumps from PPO runs.
:::

:::{grid-item-card} Scene export
:link: 4-scene_export
:link-type: doc
Export MuJoCo scenes and copied assets for inspection.
:::

::::

```{toctree}
:hidden:

1-onnx_export
2-wandb
3-nan_visualizer
4-scene_export
```
