# How-To 指南

常见任务的端到端操作指南——每一步都附带验证命令，可以边做边确认本地
检出的状态。

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 接入新机器人
:link: 1-onboard_new_robot
:link-type: doc
从 URDF/MJCF 资产到首次 PPO 训练跑通，每步都带验证。
:::

:::{grid-item-card} 添加自定义 reward term
:link: 2-custom_reward
:link-type: doc
从函数定义到在训练 run 中验证并调整权重。
:::

:::{grid-item-card} 诊断训练问题
:link: 3-training_diagnostics
:link-type: doc
NaN guard dump、reward 异常与后端不一致的排查手册。
:::

::::

## 参考速查

- 内置 reward / observation / termination term：
  {doc}`../5-reference/6-manager_terms/0-index`
- 域随机化（DR）字段：
  {doc}`../2-user_guide/5-domain_randomization/2-field_reference`
- 各算法页含配置参考表：
  {doc}`../2-user_guide/2-algorithms/0-index`

```{toctree}
:hidden:

1-onboard_new_robot
2-custom_reward
3-training_diagnostics
```
