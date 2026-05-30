# 用户指南

仓库安装完成后的日常使用参考。如果你是第一次配置 UniLab，请先阅读
{doc}`../1-getting_started/0-index`。

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 训练
:link: 1-training/0-index
:link-type: doc
CLI 路由、Hydra owner YAML、日志、检查点、Docker 以及多 GPU 说明。
:::

:::{grid-item-card} 算法
:link: 2-algorithms/0-index
:link-type: doc
对比 PPO、APPO、SAC、TD3、FlashSAC、MLX PPO、HIM-PPO 与 HORA。
:::

:::{grid-item-card} 后端
:link: 3-backends/0-index
:link-type: doc
从 owner YAML 与后端能力证据中选择 MuJoCo 或 Motrix。
:::

:::{grid-item-card} 任务
:link: 4-tasks/0-index
:link-type: doc
查找运动控制、动作追踪、操作以及移动操作任务。
:::

:::{grid-item-card} 域随机化
:link: 5-domain_randomization/0-index
:link-type: doc
通过 task owner 配置来设置 reset、init 与 interval 随机化。
:::

:::{grid-item-card} 工具
:link: 7-tooling/0-index
:link-type: doc
导出 ONNX、检查 NaN、发送 W&B 日志以及导出场景。
:::

:::{grid-item-card} 操作说明
:link: 8-manipulation/0-index
:link-type: doc
针对 Allegro、Sharpa 与 Go2+Airbot 的任务专属说明。
:::

::::

```{toctree}
:hidden:
:caption: 用户指南

1-training/0-index
2-algorithms/0-index
3-backends/0-index
4-tasks/0-index
5-domain_randomization/0-index
6-terrain/0-index
7-tooling/0-index
8-manipulation/0-index
```
