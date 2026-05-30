# Sim-to-Sim

通过 owner YAML 和 backend contract，在 MuJoCo 与 Motrix 之间迁移同一个任务。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 切换后端
:link: 1-backend_swap
:link-type: doc
通过任务 owner YAML 选择后端。
:::

:::{grid-item-card} 切换 owner YAML
:link: 2-owner_yaml_swap
:link-type: doc
为已有任务添加或检查某个后端的 YAML。
:::

:::{grid-item-card} 接触与摩擦
:link: 3-contact_and_friction_alignment
:link-type: doc
在不同模拟器之间对齐与接触相关的假设。
:::

:::{grid-item-card} Reward 一致性
:link: 4-reward_parity
:link-type: doc
检查后端边界附近的 reward 项。
:::

:::{grid-item-card} 回放差异
:link: 5-playback_and_snapshot_differences
:link-type: doc
理解渲染器与快照能力上的差异。
:::

:::{grid-item-card} 能力缺口
:link: 6-capability_gaps
:link-type: doc
用证据记录不受支持的后端特性。
:::

::::

```{toctree}
:hidden:

1-backend_swap
2-owner_yaml_swap
3-contact_and_friction_alignment
4-reward_parity
5-playback_and_snapshot_differences
6-capability_gaps
```
