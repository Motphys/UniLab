# 从 Legged Gym 迁移

语言: 简体中文

Legged Gym 曾是那套 GPU 常驻的 PPO 模板，教会了整个领域如何训练四足机器人。它的
核心思想 —— joystick 命令空间、地形课程（terrain curricula）、RSL-RL PPO ——
在 UniLab 中得以延续。因此迁移在很大程度上是机械性的。

## 直接对应关系

| Legged Gym | UniLab |
|---|---|
| `LeggedRobot` env 类 | `unilab.envs.locomotion.common.base` |
| `compute_observations()` | env 侧 obs 构建器 + `unilab.base.observations` |
| `_reward_*` 方法 | env 的 `compute_reward()` + reward 项 registry |
| `command_ranges` | 任务 owner YAML 的 `commands` 块 |
| 地形课程 | {doc}`../../2-user_guide/6-terrain/1-procedural` |
| RSL-RL PPO | `unilab.algos.torch.rsl_rl_ppo` |

## 有哪些新东西

- **两个后端。** Legged Gym 仅支持 Isaac Gym；UniLab 给你 MuJoCo + Motrix。
  在移植之前先选一个（或两个都选）；参见
  {doc}`../2-sim_to_sim/1-backend_swap`。
- **异步采集。** Legged Gym 在 GPU 上同步采集；UniLab 的
  APPO（`unilab.algos.torch.appo`）把 collector 与 learner 解耦。如果你在意
  wall-clock 时间，在建立起 reward 一致性之后，就移植到 APPO。
- **硬件部署。** Legged Gym → 真实世界部署，是各实验室各自手工搭建的流程。UniLab
  把 {doc}`../1-sim_to_real/1-overview` 流水线作为一等公民产物提供给你。

## 迁移清单

1. 把你的 URDF / MJCF asset 复制到 `src/unilab/assets/robots/<robot>/` 下。
2. 在 `src/unilab/envs/locomotion/<robot>/` 下创建一个任务模块。
3. 镜像你的 reward 项；保持名称相同，以便 reward 一致性可被 diff。
4. 翻译命令采样 —— Legged Gym 的 `_resample_commands` 在 UniLab 中变成一个
   curriculum provider。
5. 翻译地形 —— Legged Gym 的高度场生成器在 UniLab 中有一个对应物，位于
   `unilab.terrains.heightfield_terrains`。

## 验证闸门

在删除你的 Legged Gym 检出之前，先在 UniLab 中于平地上训练一个等价于 Go2 的任务，
并把 reward 项轨迹与源实现对比。如果在策略学习之前轨迹就已经分叉，那么存在 reward、
命令、复位或 DR 的不匹配；参见 {doc}`../2-sim_to_sim/4-reward_parity`。

## 另请参阅

- {doc}`1-from_isaac_lab`
- {doc}`3-from_rsl_rl`
- {doc}`5-task_config_translation`

## Navigation

- Index: [文档](0-index.md)
