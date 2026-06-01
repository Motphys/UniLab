# 契约

契约定义了任务代码、backend 适配器、runner 以及算法入口必须保持的边界。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Env 契约
:link: 1-env_contract
:link-type: doc
`NpEnvState`、reset/step 形状、观测组与 wrapper。
:::

:::{grid-item-card} Backend 契约
:link: 2-backend_contract
:link-type: doc
`SimBackend` 所有权与显式的能力支撑。
:::

:::{grid-item-card} 任务 owner 契约
:link: 3-task_owner
:link-type: doc
Hydra owner YAML 身份与 backend 选择。
:::

:::{grid-item-card} Domain randomization
:link: 4-dr_contract
:link-type: doc
Init、reset、interval 与 backend 能力边界。
:::

:::{grid-item-card} Runner 生命周期
:link: 5-runner_lifecycle
:link-type: doc
Runner 的 start/stop/checkpoint 所有权。
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
