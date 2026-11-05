# 扩展

扩展指南说明在哪里添加任务、后端、算法与地形，
同时不跨越所有权边界搬移行为。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 新任务
:link: 1-new_task
:link-type: doc
添加 env config、注册、owner YAML 与测试。
:::

:::{grid-item-card} 新后端
:link: 2-new_backend
:link-type: doc
添加 `SimBackend` 实现并声明能力。
:::

:::{grid-item-card} 新算法
:link: 3-new_algorithm
:link-type: doc
添加配置、runner 代码与脚本层组装。
:::

:::{grid-item-card} 接入新机器人
:link: 5-onboard_new_robot
:link-type: doc
端到端演练：从 URDF/MJCF 资产到首次训练跑通。
:::

:::{grid-item-card} 自定义 reward term
:link: 6-custom_reward
:link-type: doc
演练：定义、接入并在训练日志中验证一个 reward term。
:::

::::

```{toctree}
:hidden:

1-new_task
2-new_backend
3-new_algorithm
4-new_terrain
5-onboard_new_robot
6-custom_reward
```
