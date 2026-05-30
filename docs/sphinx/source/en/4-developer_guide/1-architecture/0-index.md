# Architecture

Architecture pages summarize the runtime model, ownership boundaries, scene
composition rules, and registry bootstrap contract.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: 1-overview
:link-type: doc
Runtime model, layer ownership, and validation standards.
:::

:::{grid-item-card} Runtime model
:link: 2-runtime_model
:link-type: doc
Runner lifecycle, worker/learner split, and data flow.
:::

:::{grid-item-card} Layer boundaries
:link: 3-layer_boundaries
:link-type: doc
Owner layer rules for scripts, envs, backends, and algorithms.
:::

:::{grid-item-card} Scene composition
:link: 4-scene_composition
:link-type: doc
Scene fragments, assets, and cold-path materialization.
:::

:::{grid-item-card} Registry
:link: 5-registry
:link-type: doc
Bootstrap imports and env/backend registration.
:::

::::

```{toctree}
:hidden:

1-overview
2-runtime_model
3-layer_boundaries
4-scene_composition
5-registry
```
