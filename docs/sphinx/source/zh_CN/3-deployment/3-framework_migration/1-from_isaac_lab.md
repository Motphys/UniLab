# 从 Isaac Lab 迁移

语言: 简体中文

如果你有一个想在 UniLab 中运行的 Isaac Lab 任务，本页会告诉你哪些保持不变、
哪些会改变，以及锋利的边角在哪里。

## 哪些保持不变

- Gymnasium 风格的 env 接口（`reset`、`step`、`obs/reward/info`）。
- 基于 Hydra 的配置。你现有的大部分 YAML 可以通过字段名重映射来移植。
- "任务"由 scene + reward + DR + obs 组合而成这一总体思路。
- PPO 作为默认算法 —— UniLab 开箱即带 RSL-RL 的 PPO。

## 哪些会改变

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Isaac Lab 概念
  - UniLab 对应物
  - 备注
* - `DirectRLEnv`
  - `unilab.base.np_env.NpEnv`
  - UniLab 的 obs 始终是 **dict**，而不是 tensor。
* - `RigidBody.cfg`
  - 任务侧的 asset 导入 + 场景组合
  - 参见 {doc}`../../4-developer_guide/1-architecture/4-scene_composition`。
* - GPU PhysX 后端
  - CPU MuJoCo / Motrix + GPU learner
  - 架构倒置 —— 见下文。
* - `RandomizationCfg`
  - {doc}`../../4-developer_guide/2-contracts/4-dr_contract`
  - UniLab 的 DR 只在冷路径重采样中运行。
* - `RewardManager` 链
  - env 中的 reward 组合，外加
    `unilab.training.reward` 记账
  - reward 项仍然以 key 标识，以便分量级别的日志记录。
* - `EventCfg` 事件驱动钩子
  - Phase + curriculum + DR provider
  - 钩子是显式的，而非隐式的。
```

## 架构倒置

Isaac Lab 把模拟器放在 GPU 上，让你在 PhysX 中批处理数千个 env。UniLab 把
模拟器放在 CPU 上（通常是多线程），并跨 worker **进程**做批处理，与单个 GPU
learner 共享内存。

由此带来的影响：

- 在单个 env 上，UniLab 的**每个 env 步进时间**与 Isaac 相当甚至更差。
  **吞吐量**来自进程并行 + 异步（参见 `unilab.ipc.async_runner`）。
- 你可以用 **MPS、ROCm、XPU** 作为 learner 设备 —— Isaac 仅支持 CUDA。
- 模拟器与 learner 之间**不存在 GPU 争用** —— 你的 trainer 内存占用是可预测的。

## 逐步迁移

1. **审查观测。** 确保每个观测 key 都是一个无需 GPU PhysX 查询即可表达的向量。
   如果不是，就添加一个状态估计器，或把该查询移到冷路径。
2. **移植 asset。** UniLab 以 MJCF 作为唯一真实来源（source of truth）。如果你
   有 USD，先转换为 MJCF。
3. **移植 env。** 继承 `unilab.base.np_env.NpEnv`。把 reward 计算移进 env 的
   `compute_reward()`。
4. **移植 YAML。** 按照 {doc}`5-task_config_translation` 中的表格，把 Isaac Lab
   的 `EnvCfg` 字段映射到 UniLab 任务 owner YAML。
5. **移植 reward。** 使用 {doc}`6-reward_porting` 中的食谱。
6. **验证。** 训练一个小规模运行，把 reward 曲线与你的 Isaac 基线对比。

## 你会失去什么（以及如何弥补）

- **Isaac Sim 渲染器。** 使用 Motrix 的无头视频导出，或构建一个 viser 场景
  （`unilab.visualization.viser_scene`）。
- **每个 env 的 tensor obs。** UniLab 给你的是 dict-of-arrays；如果你需要 tensor，
  用你自己的 `obs_to_tensor` 包一层。
- **内置的 GPU 侧 DR。** UniLab 的 DR 是 CPU 侧、按进程进行的。对大多数任务来说
  这已经足够；对于极端并行，使用更多的 worker 进程。

## 另请参阅

- {doc}`2-from_legged_gym`
- {doc}`3-from_rsl_rl`
- {doc}`5-task_config_translation`
- {doc}`6-reward_porting`

## Navigation

- Index: [文档](0-index.md)
