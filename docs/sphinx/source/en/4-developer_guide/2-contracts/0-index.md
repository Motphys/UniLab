# Contracts

Contracts define the boundaries that task code, backend adapters, runners, and
algorithm entrypoints must preserve.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Env contract
:link: 1-env_contract
:link-type: doc
`NpEnvState`, reset/step shape, observation groups, and wrappers.
:::

:::{grid-item-card} Backend contract
:link: 2-backend_contract
:link-type: doc
`SimBackend` ownership and explicit capability support.
:::

:::{grid-item-card} Task owner contract
:link: 3-task_owner
:link-type: doc
Hydra owner YAML identity and backend selection.
:::

:::{grid-item-card} Domain randomization
:link: 4-dr_contract
:link-type: doc
Init, reset, interval, and backend capability boundaries.
:::

:::{grid-item-card} Runner lifecycle
:link: 5-runner_lifecycle
:link-type: doc
Runner start/stop/checkpoint ownership.
:::

::::

```{toctree}
:hidden:

1-env_contract
2-backend_contract
3-task_owner
4-dr_contract
5-runner_lifecycle
```
