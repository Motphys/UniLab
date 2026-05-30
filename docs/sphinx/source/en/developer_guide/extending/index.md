# Extending

Extension guides show where to add tasks, backends, algorithms, and terrain
without moving behavior across ownership boundaries.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} New task
:link: new_task
:link-type: doc
Add env config, registration, owner YAMLs, and tests.
:::

:::{grid-item-card} New backend
:link: new_backend
:link-type: doc
Add a `SimBackend` implementation and declared capabilities.
:::

:::{grid-item-card} New algorithm
:link: new_algorithm
:link-type: doc
Add configs, runner code, and script-level assembly.
:::

:::{grid-item-card} New terrain
:link: new_terrain
:link-type: doc
Extend terrain generation on cold paths.
:::

::::

```{toctree}
:hidden:

new_task
new_backend
new_algorithm
new_terrain
```
