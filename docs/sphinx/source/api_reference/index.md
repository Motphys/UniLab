---
orphan: true
---

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
Manager-Based environment runtime and task-agnostic MDP terms layered on
top of `base`.
:::

:::{grid-item-card} 🤖 `unilab.tasks`
:link: tasks/index
:link-type: doc
Concrete locomotion, manipulation, and motion-tracking task packages.
:::

::::

## Learning stack

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🎛 Learning algorithms → `uni_rl`
:link: algos/index
:link-type: doc
PPO / APPO / SAC / TD3 variants moved to the uni_rl package (issue #1480).
:::

:::{grid-item-card} 🏋 `unilab.training`
:link: training/index
:link-type: doc
Runtime helpers, monitoring, reward bookkeeping, runner orchestration.
:::

:::{grid-item-card} 🔗 Shared-memory runtime → `uni_rl.ipc`
:link: ipc/index
:link-type: doc
Shared-memory rollout and replay primitives moved to the uni_rl package
(issue #1480).
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

:::{grid-item-card} 🧰 `unilab.utils`
:link: utils/index
:link-type: doc
Math, IO, and numerical helpers.
:::

:::{grid-item-card} 📝 Training logging → `uni_rl.logging`
:link: logging/index
:link-type: doc
W&B / TensorBoard bridges and structured logging moved to the uni_rl package
(issue #1480).
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
tasks/index
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
utils/index
logging/index
```

```{toctree}
:hidden:
:caption: Package

top_level
```
