# PPO

PPO 是默认的同步 on-policy 训练路径。它使用 `src/unilab/scripts/train_rsl_rl.py`，从
`src/unilab/conf/ppo/config.yaml` 组合配置，并直接驱动上游 RSL-RL（rsl-rl-lib）：
`unilab.rl.RslRlVecEnvAdapter` 把 env 适配到 RSL-RL 的 `VecEnv` 契约，
由 RSL-RL 自己的 `OnPolicyRunner` 训练 `rsl_rl.algorithms:PPO`。配置采用 RSL-RL v5
原生 schema —— `algo.actor:` / `algo.critic:` 模型块（如
`class_name: rsl_rl.models.MLPModel`、`hidden_dims`、`obs_normalization`），外加
`algo.algorithm:` 块（`class_name: rsl_rl.algorithms:PPO`）。

## 快速开始

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.no_play=true
```

## 常用 Override

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

使用 `uv run eval` 进行检查点回放：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

日志按 `algo.algo_log_name` 分组；`src/unilab/conf/ppo/config.yaml` 中的默认值为
`rsl_rl_ppo`。

## 单机多卡训练

`CUDA_VISIBLE_DEVICES`（每个 rank 一项）打开 RSL-RL 的同步数据并行 PPO：

```bash
export CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.no_play=true

uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  training.no_play=true
```

`CUDA_VISIBLE_DEVICES` 是唯一的 GPU 拓扑来源。父进程列表可以包含物理序号、UUID 或
MIG token；torchrun 为每个 worker 重映射成单卡可见，因此每个 rank 内部都使用本地
`cuda:0`。`training.device` 仅保留为显式单设备覆盖，不能替代 rank-local 可见性。

对 IsaacGym、IsaacSim 和 Genesis owner，同一拓扑也会传给环境仿真器。Genesis 会在
`gs.init` 前选择每个进程的 session 设备。

`algo.num_envs` 是**每个 rank** 的环境数，不是全局预算。设 rank 数为 `W`、配置
环境数为 `N`、rollout 长度为 `T`：

```text
每 rank 每轮样本数 = N * T
全局每轮样本数     = W * N * T
```

每个 rank 独立持有 env、policy copy 和 rollout storage；rollout、GAE、advantage
normalization 与 mini-batch shuffle 都留在本地。RSL-RL 在启动时广播 rank 0 的模型
状态，并在每个 PPO mini-batch backward 后平均梯度。rank `i` 使用
`algo.seed + i`。`training.num_timesteps` 按全局样本预算解释，因此推导 iteration
数时使用 `W * N * T`。

只有 rank 0 写 TensorBoard/W&B、checkpoint、`run_config.json` 与
`run_summary.json`；所有 rank 销毁 process group 后，才由 rank 0 可选进入
playback。`run_summary.json` 记录 world size、每 rank/全局环境数、每轮样本数和
聚合训练吞吐率。RSL-RL 的 `Perf/total_fps` 同样是全局 samples/s；reward 与
episode 统计仍是 rank 0 的本地视角。

RSL-RL 启动后不会同步 observation normalizer buffer 或环境 curriculum 状态。因此，
启用 observation normalization（`algo.actor.obs_normalization`，包括当前 Go2 flat owner）的 task 保留 rank-local
统计，checkpoint 保存 rank 0 的副本。这是上游 RSL-RL 的分布式语义，不是先拼接
全局 rollout 再训练的 PPO。

集成 launcher 只覆盖单机。仓库已有 `go2_joystick_flat` 与
`g1_motion_tracking` 的双卡 MuJoCo smoke；这不构成多机或 Motrix 多卡 support
声明。当前验证的 RTX 6000D 主机沿用 production NCCL 兼容默认值
`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`，用户显式环境变量优先。

## 配置参考

默认值以 `src/unilab/structured_configs.py` 中的类型化配置（`PPOConfig`）为准；
`src/unilab/conf/ppo/config.yaml` 中的取值与之相同，除非另行注明。所有字段都可以在
命令行上 override：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.algorithm.lam=0.92 \
  algo.algorithm.entropy_coef=0.005
```

### 训练效率

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 4096 | 每个 rank 的并行环境数（go2_joystick_flat/mujoco owner 设为 1024） |
| `algo.num_steps_per_env` | int | 24 | 每个环境的 rollout 长度；每轮迭代样本数 = num_envs × 该值 |
| `algo.max_iterations` | int | 101 | 训练迭代数（go2 owner 设为 151） |
| `algo.save_interval` | int | 100 | checkpoint 保存间隔（迭代数） |
| `algo.algorithm.num_learning_epochs` | int | 5 | 每批采集数据上的 SGD epoch 数 |
| `algo.algorithm.num_mini_batches` | int | 4 | 每个 epoch 的 mini-batch 数 |
| `algo.seed` | int | 1 | 基础随机种子；rank `i` 使用 `seed + i` |
| `training.device` | str / null | null | 显式单设备覆盖；null = 自动放置 |
| `training.log_interval` | int | 1 | TensorBoard/W&B 标量上报间隔（迭代数） |

### 策略性能

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.algorithm.gamma` | float | 0.99 | 折扣因子 |
| `algo.algorithm.lam` | float | 0.95 | GAE lambda；调小以偏差换方差 |
| `algo.algorithm.entropy_coef` | float | 0.01 | 熵奖励系数（go2 owner 降为 1e-3） |
| `algo.algorithm.learning_rate` | float | 1e-3 | Adam 学习率；adaptive schedule 下为初始值（go2 owner 设为 3e-4） |
| `algo.algorithm.schedule` | str | adaptive | 学习率 schedule；`adaptive` 将 KL 拉向 `desired_kl` |
| `algo.algorithm.desired_kl` | float | 0.01 | adaptive 学习率 schedule 的 KL 目标 |
| `algo.algorithm.clip_param` | float | 0.2 | PPO surrogate 裁剪范围 |
| `algo.algorithm.value_loss_coef` | float | 1.0 | value loss 项的权重 |
| `algo.algorithm.use_clipped_value_loss` | bool | true | 像策略目标一样裁剪 value target |
| `algo.algorithm.max_grad_norm` | float | 1.0 | 梯度范数裁剪 |
| `algo.actor.hidden_dims` | list[int] | [512, 256, 128] | 策略 MLP 各层宽度 |
| `algo.actor.activation` | str | elu | 策略 MLP 激活函数 |
| `algo.actor.obs_normalization` | bool | false | 归一化策略观测（go2 owner 开启） |
| `algo.actor.distribution_cfg.init_std` | float | 1.0 | 高斯动作初始标准差（go2 owner 设为 0.5） |
| `algo.critic.hidden_dims` | list[int] | [512, 256, 128] | Critic MLP 各层宽度 |
| `algo.critic.obs_normalization` | bool | false | 归一化 critic 观测（go2 owner 开启） |

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
| `training.num_timesteps` | int / null | null | 全局样本预算；设置后覆盖推导出的迭代数 |

### Reward 权重

Reward 项位于任务 owner YAML 的 `reward:` 段 —— 每项一个块，含 `func`、`weight`
与 `params`（示例：`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`）。调整某个
项可以编辑该 owner YAML，也可以在命令行覆盖其权重：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  reward.tracking_lin_vel.weight=1.5 \
  reward.action_rate.weight=-0.01
```

各项名称与默认权重因任务而异；请查看所训练任务 owner YAML 的 `reward:` 段
（`src/unilab/conf/ppo/task/<task>/base.yaml`）。
