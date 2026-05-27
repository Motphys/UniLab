# Developer Guide

Submitting a PR, extending UniLab with a new backend / task / algorithm, or
auditing the runtime architecture? Start here.

## Onboarding

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🤝 Contributing
:link: contributing
:link-type: doc
Dev environment, code style, tests, PR template.
:::

:::{grid-item-card} 🔄 PR workflow
:link: contributing_workflow
:link-type: doc
Branching, ADR-first changes, review etiquette.
:::

::::

## Architecture

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 📐 Development standard
:link: architecture/development_standard
:link-type: doc
:::

:::{grid-item-card} 🏗 Runtime model
:link: architecture/runtime_model
:link-type: doc
:::

:::{grid-item-card} 🧱 Layer boundaries
:link: architecture/layer_boundaries
:link-type: doc
:::

:::{grid-item-card} 🎬 Scene composition
:link: architecture/scene_composition
:link-type: doc
:::

:::{grid-item-card} 🧷 Registry bootstrap
:link: architecture/registry_bootstrap
:link-type: doc
:::

::::

## Contracts

These are the *load-bearing* interfaces. Break one and downstream tasks break.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 📋 NpEnv contract
:link: contracts/env_contract
:link-type: doc
:::

:::{grid-item-card} 🔌 Backend capability
:link: contracts/backend_capability
:link-type: doc
:::

:::{grid-item-card} ⚙ Task owner config
:link: contracts/task_owner_config
:link-type: doc
:::

:::{grid-item-card} 🎲 Domain randomization
:link: contracts/domain_randomization
:link-type: doc
:::

:::{grid-item-card} 🔁 Runner lifecycle
:link: contracts/runner_lifecycle
:link-type: doc
:::

::::

## Extending UniLab

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🆕 New task
:link: extending/new_task
:link-type: doc
:::

:::{grid-item-card} 🧮 New backend
:link: extending/new_backend
:link-type: doc
:::

:::{grid-item-card} 🤖 New algorithm
:link: extending/new_algorithm
:link-type: doc
:::

:::{grid-item-card} 🏞 New terrain
:link: extending/new_terrain
:link-type: doc
:::

::::

## Architecture Decision Records

```{toctree}
:maxdepth: 1

adr/README
adr/ADR-0000-index
adr/ADR-0001-runtime-model-and-layer-boundaries
adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot
adr/ADR-0003-task-owner-and-config-compose-contract
adr/ADR-0004-registry-bootstrap-contract
adr/ADR-0005-unified-obs-critic-env-and-ipc-contract
adr/ADR-TEMPLATE
```

## 中文文档(zh_CN)

```{toctree}
:maxdepth: 1
:caption: 中文 Developer 文档

zh_CN/development-standard
zh_CN/collaboration
zh_CN/CONTRIBUTING
zh_CN/domain-randomization-contract
zh_CN/scene-composition-design
zh_CN/motion-asset-migration
zh_CN/motrix-contact-sensor-notes
```

```{toctree}
:hidden:
:caption: Architecture

architecture/development_standard
architecture/runtime_model
architecture/layer_boundaries
architecture/scene_composition
architecture/registry_bootstrap
```

```{toctree}
:hidden:
:caption: Contracts

contracts/env_contract
contracts/backend_capability
contracts/task_owner_config
contracts/domain_randomization
contracts/runner_lifecycle
```

```{toctree}
:hidden:
:caption: Extending

extending/new_task
extending/new_backend
extending/new_algorithm
extending/new_terrain
```

```{toctree}
:hidden:
:caption: Onboarding

contributing
contributing_workflow
```
