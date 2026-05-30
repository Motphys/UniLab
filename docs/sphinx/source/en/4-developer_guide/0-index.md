# Developer Guide

Use this section when you are changing UniLab itself: runtime contracts,
backend capabilities, task owners, algorithms, tests, or contribution workflow.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Architecture overview
:link: 1-architecture/1-overview
:link-type: doc
Runtime model, layer ownership, config-first rules, and validation standards.
:::

:::{grid-item-card} Registry
:link: 1-architecture/5-registry
:link-type: doc
Bootstrap imports, env registration, and runtime construction.
:::

:::{grid-item-card} Env contract
:link: 2-contracts/1-env_contract
:link-type: doc
`NpEnvState`, reset/step shape, observation groups, and wrapper expectations.
:::

:::{grid-item-card} Backend contract
:link: 2-contracts/2-backend_contract
:link-type: doc
The `SimBackend` boundary and optional capability pattern.
:::

:::{grid-item-card} Task owner contract
:link: 2-contracts/3-task_owner
:link-type: doc
Hydra owner YAML identity and backend-selection rules.
:::

:::{grid-item-card} Domain randomization contract
:link: 2-contracts/4-dr_contract
:link-type: doc
Init, reset, interval, and backend capability boundaries for DR providers.
:::

::::

## Extending

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} New task
:link: 3-extending/1-new_task
:link-type: doc
Add env config, registration, owner YAMLs, and tests.
:::

:::{grid-item-card} New backend
:link: 3-extending/2-new_backend
:link-type: doc
Add a `SimBackend` implementation and explicit capability support.
:::

:::{grid-item-card} New algorithm
:link: 3-extending/3-new_algorithm
:link-type: doc
Add configs, runner code, and script-level assembly without changing env contracts.
:::

:::{grid-item-card} New terrain
:link: 3-extending/4-new_terrain
:link-type: doc
Extend terrain generation while keeping asset access on cold paths.
:::

::::

## Contributor Workflow

- {doc}`4-contributing`
- {doc}`5-contributing_workflow`
- {doc}`6-agent_quick_reference`
- {doc}`ADR index </adr/ADR-0000-index>`

```{toctree}
:hidden:
:caption: Developer Guide

1-architecture/0-index
2-contracts/0-index
3-extending/0-index
4-contributing
5-contributing_workflow
6-agent_quick_reference
```
