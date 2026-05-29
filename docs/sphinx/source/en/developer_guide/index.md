# Developer Guide

Use this section when you are changing UniLab itself: runtime contracts,
backend capabilities, task owners, algorithms, tests, or contribution workflow.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Architecture overview
:link: architecture/overview
:link-type: doc
Runtime model, layer ownership, config-first rules, and validation standards.
:::

:::{grid-item-card} Registry
:link: architecture/registry
:link-type: doc
Bootstrap imports, env registration, and runtime construction.
:::

:::{grid-item-card} Env contract
:link: contracts/env_contract
:link-type: doc
`NpEnvState`, reset/step shape, observation groups, and wrapper expectations.
:::

:::{grid-item-card} Backend contract
:link: contracts/backend_contract
:link-type: doc
The `SimBackend` boundary and optional capability pattern.
:::

:::{grid-item-card} Task owner contract
:link: contracts/task_owner
:link-type: doc
Hydra owner YAML identity and backend-selection rules.
:::

:::{grid-item-card} Domain randomization contract
:link: contracts/dr_contract
:link-type: doc
Init, reset, interval, and backend capability boundaries for DR providers.
:::

::::

## Extending

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} New task
:link: extending/new_task
:link-type: doc
Add env config, registration, owner YAMLs, and tests.
:::

:::{grid-item-card} New backend
:link: extending/new_backend
:link-type: doc
Add a `SimBackend` implementation and explicit capability support.
:::

:::{grid-item-card} New algorithm
:link: extending/new_algorithm
:link-type: doc
Add configs, runner code, and script-level assembly without changing env contracts.
:::

:::{grid-item-card} New terrain
:link: extending/new_terrain
:link-type: doc
Extend terrain generation while keeping asset access on cold paths.
:::

::::

## Contributor Workflow

- {doc}`contributing`
- {doc}`contributing_workflow`
- {doc}`agent_quick_reference`
- {doc}`ADR index </adr/ADR-0000-index>`

```{toctree}
:hidden:
:caption: Architecture

architecture/overview
architecture/runtime_model
architecture/layer_boundaries
architecture/scene_composition
architecture/registry
```

```{toctree}
:hidden:
:caption: Contracts

contracts/env_contract
contracts/backend_contract
contracts/task_owner
contracts/dr_contract
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
agent_quick_reference
```
