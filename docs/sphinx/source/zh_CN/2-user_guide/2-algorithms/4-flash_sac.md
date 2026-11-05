# FlashSAC

FlashSAC 通过 `src/unilab/scripts/train_flashsac.py` 运行，拥有独立的配置树。使用
`--algo flashsac` 选择它；默认值内联在 `src/unilab/conf/flashsac/config.yaml` 中，实现位于
`unilab.rl.algos.flash_sac` 下。

它与 SAC 共用 off-policy runner 设计，但默认网络并不相同：actor 使用
block-based 结构，critic 使用 distributional（categorical）Q 变体。

## 快速开始

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## 关键字段

对于 off-policy 回放路径（`src/unilab/scripts/train_flashsac.py` / CLI `--algo flashsac`），设
置 `training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。
参见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

## Learner 加速开关

单 NVIDIA learner 默认开启全部三层加速，但它们是三个独立的配置选择：

- `algo.algo_params.use_compile` 控制 CUDA Inductor 编译。设为 `false` 后走
  eager 的逐 update 兼容路径，同时也会关闭依赖编译的 whole-cycle graph。
- `algo.algo_params.compile_full_objectives` 选择编译完整 actor/critic objective，
  还是较小的 loss-tensor 区域；它不控制 CUDA Graph 捕获。
- `algo.algo_params.use_whole_cycle_cuda_graph` 控制 learner 自己捕获并回放整轮
  CUDA Graph。只把此字段设为 `false`，可以保留 Inductor 编译而关闭整轮捕获。

例如，使用 eager 调试路径：

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  algo.algo_params.use_compile=false
```

FlashSAC 要求同步采集，并与 SAC 共用唯一 replay 路径：有界 host ingress 加
一个驻留在 CUDA 或 Apple MPS learner device 上的完整 replay ring。CPU 与 XPU
training 不受支持。

日志根目录为 `logs/flash_sac/<task>/`。

## Checkpoint 兼容性

每个 FlashSAC checkpoint 都会记录 `actor_semantics` 载荷
（`schema_version`/`kind`/`builder`/`kwargs`/`obs_dim`/`action_dim`/
`action_transform`，由 `unilab.rl.algos.flash_sac.actor_spec` 写入）。训练
续训与 play 两条加载路径都会在恢复任何权重之前对其 fail-closed 校验，内置
actor 与经 `algo.algo_params.actor_config` 注入的自定义 actor 同样适用。

pre-contract checkpoint（缺失或为 null 的 `actor_semantics` 条目）对所有
actor 一律不再支持，包括内置 actor：加载会 fail closed，错误信息说明最低
支持的格式。唯一恢复路径是使用当前版本重新训练策略。支持基线从 1.3.5 之后
首个包含该变更的版本开始，不回 port 到任何 1.3.5 patch release。

## 多卡数据并行

FlashSAC 与 SAC 共用同一套多卡数据并行机制：`CUDA_VISIBLE_DEVICES` 下每个 rank 各跑一
套独立的 learner+collector；启动时广播完整模型状态，稳态在每个实际 optimizer step
前分别平均 actor / critic / temperature 梯度。仅 rank 0 保存 checkpoint。用法与限制见
{doc}`/zh_CN/2-user_guide/2-algorithms/3-sac` 的"多卡数据并行"小节。

```bash
export CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  training.no_play=true
```

## 配置参考

默认值以 `src/unilab/structured_configs.py` 中的类型化配置（`FlashSACConfig`）为准；
`src/unilab/conf/flashsac/config.yaml` 中的取值与之相同，除非另行注明。所有字段都可以
在命令行上 override：

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  algo.gamma=0.99 \
  algo.algo_params.temp_initial_value=0.1
```

### 训练效率

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 1024 | 并行采集环境数 |
| `algo.batch_size` | int | 2048 | 每次 learner 更新的 batch |
| `algo.replay_buffer_n` | int | 512 | replay 容量（以整轮 env 向量为单位；行数 = n × num_envs） |
| `algo.updates_per_step` | int | 2 | 每个 env step 的 learner 更新数 |
| `algo.learning_starts` | int | 98 | learner 开始更新前需采集的整轮 env 向量数 |
| `algo.policy_frequency` | int | 2 | 每 N 次 learner 更新做一次 actor/温度更新 |
| `algo.max_iterations` | int | 5000 | 训练迭代数 |
| `algo.save_interval` | int | 1000 | checkpoint 保存间隔（迭代数） |
| `algo.algo_params.use_compile` | bool | true | CUDA Inductor 编译；`false` 走 eager 路径并同时关闭 whole-cycle graph |
| `algo.algo_params.compile_full_objectives` | bool | true | 编译完整 actor/critic objective，而非较小的 loss-tensor 区域 |
| `algo.algo_params.use_whole_cycle_cuda_graph` | bool | true | learner 侧整轮 CUDA Graph 捕获 |
| `algo.algo_params.dp_whole_cycle_cuda_graph` | bool | true | 数据并行路径上的整轮捕获；`false` 只关闭 DP 捕获（该字段设于 `config.yaml`） |
| `algo.seed` | int | 1 | 随机种子 |
| `training.use_amp` | bool | true | learner 更新路径上的 bf16 autocast |
| `training.log_interval` | int | 1 | TensorBoard/W&B 标量上报间隔（迭代数） |

### 策略性能

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.gamma` | float | 0.97 | 折扣因子 |
| `algo.tau` | float | 0.01 | target critic 的 EMA 系数 |
| `algo.actor_lr` | float | 3e-4 | 兜底峰值 LR（当 `algo.algo_params.learning_rate_peak` ≤ 0 时使用） |
| `algo.critic_lr` | float | 3e-4 | 声明的 LR 字段；实际 schedule 由 `learning_rate_*` 决定 |
| `algo.actor_hidden_dim` | int | 128 | actor 网络宽度 |
| `algo.critic_hidden_dim` | int | 256 | critic 网络宽度 |
| `algo.num_atoms` | int | 101 | 分布式 critic 的 atom 数 |
| `algo.obs_normalization` | bool | false | 归一化观测 |
| `algo.use_layer_norm` | bool | false | 网络中使用 LayerNorm |
| `algo.policy_before_critic` | bool | false | 每次更新先更新 actor（+温度）再更新 critic（该字段设于 `config.yaml`） |
| `algo.algo_params.normalize_reward` | bool | true | 用运行中的回报估计归一化 reward |
| `algo.algo_params.normalized_g_max` | float | 5.0 | 归一化回报估计的截断上限 |
| `algo.algo_params.actor_num_blocks` | int | 2 | block 结构 actor 的块数 |
| `algo.algo_params.critic_num_blocks` | int | 2 | block 结构 critic 的块数 |
| `algo.algo_params.actor_bc_alpha` | float | 0.0 | 行为克隆辅助损失权重；0 表示关闭 |
| `algo.algo_params.actor_noise_zeta_mu` | float | 2.0 | actor 噪声调度的 mu |
| `algo.algo_params.actor_noise_zeta_max` | int | 16 | actor 噪声调度的上限 |
| `algo.algo_params.critic_min_v` | float | -5.0 | 分布式 critic 值域下界 |
| `algo.algo_params.critic_max_v` | float | 5.0 | 分布式 critic 值域上界 |
| `algo.algo_params.temp_initial_value` | float | 0.01 | 初始熵温度 |
| `algo.algo_params.temp_target_sigma` | float | 0.15 | 温度自适应使用的目标 sigma |
| `algo.algo_params.temp_target_entropy` | float / null | null | 显式目标熵；null 使用内置目标 |
| `algo.algo_params.learning_rate_init` | float | 3e-4 | actor/critic/温度调度起始 LR |
| `algo.algo_params.learning_rate_peak` | float | 3e-4 | 调度峰值 LR |
| `algo.algo_params.learning_rate_end` | float | 1.5e-4 | 衰减结束后的 LR |
| `algo.algo_params.learning_rate_warmup_steps` | int | 0 | 从起始值到峰值的 warmup 步数 |
| `algo.algo_params.learning_rate_decay_steps` | int | 500000 | 从峰值到结束值的衰减步数 |
| `algo.algo_params.n_step` | int | 1 | N-step 回报长度；当前仅支持 1 |
| `algo.algo_params.amp_dtype` | str | auto | `training.use_amp` 下的 AMP dtype |

### 训练运行（`training:` 段）

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `training.no_play` | bool | false | 训练结束后跳过策略回放 |
| `training.play_only` | bool | false | 跳过训练，只回放已有 checkpoint |
| `training.export_onnx` | bool | true | 回放时导出 `policy.onnx` |
| `training.cuda_process_sharing` | str / null | null | null = 独立 CUDA context；`mps` 请求 CUDA MPS（守护进程由用户经 `uni-cumps` 管理） |
| `training.dp_gradient_wire_dtype` | str | float32 | 多卡梯度平均的传输 dtype；`bfloat16` 会压缩 |
| `training.nan_guard.enabled` | bool | true | 检测非有限张量并转储诊断信息 |
| `training.nan_guard.check_interval` | int | 32 | NaN 检查之间的 env step 数 |
| `training.defer_finite_validation` | bool | true | 将有限性检查合并为周期性批次 |
| `training.finite_validation_interval` | int | 32 | 每个批量有限性检查周期的步数 |
| `training.collector_tensor_device` | str | auto | collector 侧张量设备；`auto` 跟随 learner |
| `training.env_steps_per_sync` | int | 1 | 每次采集同步的 env step 数 |
| `training.inference_slot_capacity` | int | 1 | 在途 learner 推理请求数；1 表示严格串行 |
| `training.collector_metrics_interval` | int | 32 | collector 指标刷写间隔（步数） |
| `training.replay_ingress_depth` | int | 2 | 允许在途的 replay ingress 块数 |
| `training.replay_ingress_slot_rows` | int / null | null | 每个 ingress 块的行数；null = 每个 collector 向量一块 |
| `training.replay_prefetch_mode` | str | one_tick | replay 预取策略 |
| `training.trace_enabled` | bool | false | off-policy 运行时的 Chrome trace 剖析 |
| `training.trace_output_dir` | str / null | null | trace 输出目录 |
| `training.trace_thread_time` | bool | false | trace 中包含线程时间 |
| `training.trace_cuda_events` | bool | true | trace 中记录 CUDA event |
| `training.nvtx_profile_ranges` | bool | false | 为外部剖析器发出 NVTX range |
| `training.torch_threads.enabled` | bool | true | 按进程角色固定 torch CPU 线程数 |
| `training.torch_threads.learner_num_threads` | int / str | auto | learner 的 intra-op 线程数；`auto` 按主机 CPU 数解析 |
| `training.torch_threads.collector_num_threads` | int / str | auto | collector 的 intra-op 线程数 |
| `training.dp_collector_cpu_ids` | list / null | null | 每个 rank 的显式 CPU id；null 按 `cpu_count // world_size` 自动划分 |
| `training.logger` | str | tensorboard | 标量日志后端（`tensorboard` 或 `wandb`） |

### Reward 权重

Reward 项位于任务 owner YAML 的 `reward:` 段 —— 每项一个块，含 `func`、`weight`
与 `params`（示例：`src/unilab/conf/flashsac/task/g1_walk_flat/base.yaml`）。调整某个
项可以编辑该 owner YAML，也可以在命令行覆盖其权重：

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  reward.tracking_lin_vel.weight=2.5
```

各项名称与默认权重因任务而异；请查看所训练任务 owner YAML 的 `reward:` 段
（`src/unilab/conf/flashsac/task/<task>/base.yaml`）。
