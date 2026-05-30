# 框架迁移

把已有的任务或训练流程，从相邻的 RL 框架迁移进 UniLab 的 contract 驱动布局。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 从 Isaac Lab 迁移
:link: 1-from_isaac_lab
:link-type: doc
把 GPU 常驻的任务结构映射到 UniLab 的 CPU sim 与 learner 拆分。
:::

:::{grid-item-card} 从 Legged Gym 迁移
:link: 2-from_legged_gym
:link-type: doc
把基于类的环境迁移到 `NpEnv` contract。
:::

:::{grid-item-card} 从 RSL-RL 迁移
:link: 3-from_rsl_rl
:link-type: doc
把 trainer 的假设与 UniLab 的 runner 组装分离开。
:::

:::{grid-item-card} 从 skrl 迁移
:link: 4-from_skrl
:link-type: doc
映射算法入口与配置归属。
:::

:::{grid-item-card} 配置翻译
:link: 5-task_config_translation
:link-type: doc
对照各配置中常见字段的归属关系。
:::

:::{grid-item-card} Reward 移植
:link: 6-reward_porting
:link-type: doc
在不破坏 env/backend contract 的前提下移植 reward 项。
:::

::::

```{toctree}
:hidden:

1-from_isaac_lab
2-from_legged_gym
3-from_rsl_rl
4-from_skrl
5-task_config_translation
6-reward_porting
```
