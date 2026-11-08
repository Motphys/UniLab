# APPO

APPO 是 UniLab 的异步 PPO 路径。它使用 `src/unilab/scripts/train_appo.py`、
`src/unilab/conf/appo/config.yaml` 以及 `unilab.rl.algos.appo` 下的运行时。该配置暴露
了 `algo.steps_per_env`、`training.collector_device` 和
`training.replay_queue_size`；算法配置中包含 V-trace 裁剪字段。

## 快速开始

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
```

## 常用 Override

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.replay_queue_size=2
```

回放与检查点选择使用 `uv run eval`：

```bash
uv run eval --algo appo --task go2_joystick_flat --sim mujoco --load-run -1
```

## 运行模型

- collector 负责 CPU 仿真，learner 负责 GPU 训练。
- rollout 会先进入 replay queue，再由 learner 消费。
- APPO 内部带 V-trace importance-sampling 修正，更新语义不同于同步 PPO。
- collector / learner 流水线由一个 4 槽 ring buffer 支撑。

单次迭代的计时时序（各指标含义见[日志页](../1-training/3-logging.md)）：

```{mermaid}
gantt
    title 一次 Learner 迭代的时间线（APPO）
    dateFormat x
    axisFormat %S

    section Collector（进程）
    rollout N · env interaction（mlp_infer + env_step）×steps_per_env :active, c0, 0, 12000
    rollout N+1（与 learner 并行采集）                                :active, c1, 13000, 30000

    section Ring Buffer（4 槽）
    rollout N 就绪    :milestone, r0, 12000, 12000
    rollout N+1 就绪  :milestone, r1, 30000, 30000

    section Learner（GPU）
    Collector Wait（缓冲满则约 0）    :done,   l0, 12000, 13000
    Replay Stage（ring 进 staging）   :        l1, 13000, 15500
    Replay Sample（组合 batch view）  :        l2, 15500, 16000
    Train（V-trace + PPO SGD）        :active, l3, 16000, 28000
    Weight Publish 写回 collector     :crit,   l4, 28000, 30000

    section Iter Wall
    Perf/iteration_time（仅 learner 这圈）  :        l5, 12000, 30000
```

> 横轴为示意相对时长（非真实 ms 比例）。collector 子进程经 4 槽 ring buffer 与 learner 并行产出 rollout，稳态下 **Collector Wait ≈ 0**。`Perf/iteration_time` 仅计 learner 这一圈（含 Collector Wait，但不含 collector 的并行采集计算）；红色 Weight Publish 标志该轮迭代结束、向 collector 发布新权重。各字段含义见[日志页](../1-training/3-logging.md)。

## 关键字段

- `algo.steps_per_env`：单个环境的 rollout 长度。
- `training.replay_queue_size`：learner 侧缓存深度。
- `training.collector_device`：collector 设备；默认跟随 learner。
- `algo.save_interval`：checkpoint 保存间隔。

默认日志根目录为 `logs/appo/<task>/`，来自 `src/unilab/conf/appo/config.yaml` 中的
`algo.algo_log_name=appo`。

## 配置参考

默认值以 `src/unilab/structured_configs.py` 中的类型化配置（`APPOConfig`）为准；
`src/unilab/conf/appo/config.yaml` 中的取值与之相同，除非另行注明。所有字段都可以在
命令行上 override：

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.algorithm.lam=0.92 \
  algo.algorithm.vtrace_clip_rho=1.2
```

### 训练效率

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 2048 | 每个 collector 的环境数 |
| `algo.steps_per_env` | int | 24 | 每份发布的 rollout 中单个环境的长度 |
| `algo.max_iterations` | int | 150 | learner 迭代数 |
| `algo.save_interval` | int | 50 | checkpoint 保存间隔（迭代数） |
| `algo.algorithm.num_learning_epochs` | int | 5 | 每份消费的 rollout 上的 SGD epoch 数 |
| `algo.algorithm.num_mini_batches` | int | 4 | 每个 epoch 的 mini-batch 数 |
| `algo.algorithm.enable_compile` | bool | true | learner 更新的 torch.compile；内置 `src/unilab/conf/appo/config.yaml` 出货为 `false` |
| `algo.seed` | int | 1 | 随机种子 |
| `training.collector_device` | str / null | null | collector 设备；null 跟随 learner |
| `training.replay_queue_size` | int / null | null | learner 侧 replay 缓存深度 |
| `training.log_interval` | int | 1 | TensorBoard/W&B 标量上报间隔（迭代数） |

### 策略性能

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.algorithm.gamma` | float | 0.99 | 折扣因子 |
| `algo.algorithm.lam` | float | 0.95 | GAE lambda；调小以偏差换方差 |
| `algo.algorithm.entropy_coef` | float | 0.01 | 熵奖励系数 |
| `algo.algorithm.learning_rate` | float | 1e-3 | Adam 学习率；adaptive schedule 下为初始值 |
| `algo.algorithm.schedule` | str | adaptive | 学习率 schedule；`adaptive` 将 KL 拉向 `desired_kl` |
| `algo.algorithm.desired_kl` | float | 0.01 | adaptive 学习率 schedule 的 KL 目标 |
| `algo.algorithm.adaptive_kl_factor` | float | 1.2 | KL 死区：高于 `desired_kl` × 该值时 LR 下降，低于 `desired_kl` ÷ 该值时 LR 上升 |
| `algo.algorithm.adaptive_lr_factor` | float | 1.1 | adaptive schedule 的 LR 乘性步长 |
| `algo.algorithm.clip_param` | float | 0.2 | PPO surrogate 裁剪范围 |
| `algo.algorithm.value_loss_coef` | float | 1.0 | value loss 项的权重 |
| `algo.algorithm.use_clipped_value_loss` | bool | true | 像策略目标一样裁剪 value target |
| `algo.algorithm.max_grad_norm` | float | 1.0 | 梯度范数裁剪 |
| `algo.algorithm.optimizer` | str | adam | learner 更新使用的优化器 |
| `algo.algorithm.vtrace_clip_rho` | float | 1.0 | V-trace 重要性权重裁剪 ρ̄ |
| `algo.algorithm.vtrace_clip_c` | float | 1.0 | V-trace 重要性权重裁剪 c̄ |
| `algo.algorithm.tau` | float | 1.0 | target network 的 EMA 系数（1.0 = 硬拷贝） |
| `algo.algorithm.target_update_freq` | int | 1 | 两次 target network 同步之间的 learner 更新数 |
| `algo.actor.hidden_dims` | list[int] | [512, 256, 128] | 策略 MLP 各层宽度 |
| `algo.actor.activation` | str | elu | 策略 MLP 激活函数 |
| `algo.actor.distribution_cfg.init_std` | float | 1.0 | 高斯动作初始标准差 |
| `algo.critic.hidden_dims` | list[int] | [512, 256, 128] | Critic MLP 各层宽度 |

### 训练运行（`training:` 段）

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `training.no_play` | bool | false | 训练结束后跳过策略回放 |
| `training.play_only` | bool | false | 跳过训练，只回放已有 checkpoint |
| `training.tf32_matmul` | str | ieee | learner FP32 矩阵乘后端；`tf32` 启用 tensor core 并自动回退 IEEE |
| `training.nan_guard.enabled` | bool | true | 检测非有限张量并转储诊断信息 |
| `training.nan_guard.check_interval` | int | 32 | NaN 检查之间的 env step 数 |
| `training.defer_finite_validation` | bool | true | 将有限性检查合并为周期性批次 |
| `training.finite_validation_interval` | int | 32 | 每个批量有限性检查周期的步数 |
| `training.collector_tensor_device` | str | auto | collector 侧张量设备；`auto` 跟随 learner |
| `training.logger` | str | tensorboard | 标量日志后端（`tensorboard` 或 `wandb`） |

### Reward 权重

Reward 项位于任务 owner YAML 的 `reward:` 段 —— 每项一个块，含 `func`、`weight`
与 `params`（示例：`src/unilab/conf/appo/task/go2_joystick_flat/base.yaml`）。调整某个
项可以编辑该 owner YAML，也可以在命令行覆盖其权重：

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  reward.tracking_lin_vel.weight=1.5 \
  reward.action_rate.weight=-0.01
```

各项名称与默认权重因任务而异；请查看所训练任务 owner YAML 的 `reward:` 段
（`src/unilab/conf/appo/task/<task>/base.yaml`）。
