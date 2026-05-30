# API Reference

Detailed reference for the `unilab` Python package. Every public class,
function, and submodule is auto-extracted from the source tree via
`sphinx.ext.autodoc` + `autosummary`. If a symbol you expect is missing,
please [open an issue](https://github.com/unilabsim/UniLab/issues).

```{note}
Reading this in a *preview* build that says "no API reference"? The build
environment couldn't import `unilab`. Install UniLab in the same venv, or
read this section on the live site.
```

## Core foundations

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🧱 `unilab.base`
:link: base/index
:link-type: doc
The contracts everything else depends on: `NpEnv`, `SimBackend`, `Registry`,
`Scene`.
:::

:::{grid-item-card} 🧪 `unilab.envs`
:link: envs/index
:link-type: doc
Concrete tasks — locomotion, manipulation, motion tracking — layered on
top of `base`.
:::

::::

## Learning stack

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🎛 `unilab.algos`
:link: algos/index
:link-type: doc
PPO / APPO / SAC / TD3 variants, in PyTorch and MLX flavours.
:::

:::{grid-item-card} 🏋 `unilab.training`
:link: training/index
:link-type: doc
Runtime helpers, monitoring, reward bookkeeping, runner orchestration.
:::

:::{grid-item-card} 🔗 `unilab.ipc`
:link: ipc/index
:link-type: doc
Shared-memory rollout and replay primitives that connect CPU workers and
the GPU learner.
:::

:::{grid-item-card} 🧮 `unilab.backend`
:link: backend/index
:link-type: doc
MuJoCo and Motrix adapters that implement `SimBackend`.
:::

::::

## Subsystems

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🎲 `unilab.dr`
:link: dr/index
:link-type: doc
Declarative domain randomization manager.
:::

:::{grid-item-card} 🏞 `unilab.terrains`
:link: terrains/index
:link-type: doc
Procedural and heightfield terrain generators.
:::

:::{grid-item-card} 👁 `unilab.visualization`
:link: visualization/index
:link-type: doc
Scene rendering and viser bridges.
:::

:::{grid-item-card} 🔧 `unilab.tools`
:link: tools/index
:link-type: doc
Scene export, NaN visualizer, ONNX export.
:::

:::{grid-item-card} 🧰 `unilab.utils`
:link: utils/index
:link-type: doc
Math, IO, and numerical helpers.
:::

:::{grid-item-card} 📝 `unilab.logging`
:link: logging/index
:link-type: doc
W&B / TensorBoard bridges and structured logging.
:::

::::

## Top-level package

```{toctree}
:maxdepth: 1

top_level
```

```{toctree}
:hidden:
:caption: Core

base/index
envs/index
```

```{toctree}
:hidden:
:caption: Learning stack

algos/index
training/index
ipc/index
backend/index
```

```{toctree}
:hidden:
:caption: Subsystems

dr/index
terrains/index
visualization/index
tools/index
utils/index
logging/index
```

```{toctree}
:hidden:
:caption: Package

top_level
```
