# SAC

SAC 通过 `src/unilab/scripts/train_sac.py` 运行；FlashSAC 有独立的入口与按算法
划分的配置树。主配置为 `src/unilab/conf/sac/config.yaml`，SAC 算法的默认值内联在其中。
当前的日志名称为 `sac`。

## 运行模型

off-policy runner 通过有界 shared memory 把仿真采集与 accelerator 学习解耦。
collector 子进程通过两个 packed ingress slot 发布 transition，完整 replay ring 只由
一个 CUDA 或 Apple MPS learner device 持有。因此 host replay 分配不随 replay
capacity 增长；slot 的 device copy 完成后才推进 `ptr` 与 `size`。CUDA 通过 side
stream 提交和 commit，MPS 的 device work 只由 learner 线程提交，不使用后台 Metal
submission。CPU 与 XPU training 不受支持，也不存在第二套 replay pipeline。

## 快速开始

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

## 关键字段

对于 off-policy 回放路径（`src/unilab/scripts/train_sac.py` / CLI `--algo sac`），设置
`training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。参
见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` 是 learner 每次 update 的 batch。
- `algo.max_iterations=500`
- 共享 off-policy 配置中的 `training.use_amp=true`

off-policy device replay 路径统一使用同步、learner-owned inference：collector 通过
shared memory 交换 observation/action，不持有 actor。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```

## 多卡数据并行

`CUDA_VISIBLE_DEVICES` 是唯一 GPU 拓扑来源：父进程列表中每个 opaque 条目对应一个
rank，rank 内 learner 与 collector 共同使用本地 `cuda:0`。rank 0 负责 spawn 其余
rank 子进程。启动时 rank 0 一次性广播 actor、critic、target critic 和温度状态；稳态
训练不再平均参数，而是在每个实际 optimizer step 前对对应的 actor、critic 或温度梯度
执行阻塞式 flat-gradient `all_reduce(SUM) / world_size`。各 rank 不交换 replay 数据，
在相同初值、平均梯度和更新顺序下各自维护一致的 optimizer 状态。

每个 rank 当前只创建一个 collector。每个 rank 都是单卡可见，因此 probe env、
collector 正式 env、replay、inference ring 与 learner 都落在同一张物理卡上。runtime manifest 的 `collector_backend_device` 记录本 rank 的
实际绑定。

off-policy 不再有设备字段，GPU 由进程可见性决定。单 rank 只需导出单卡可见；
多 rank 由父进程导出与 rank 数相同的 opaque 条目：

```bash
export CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.no_play=true
```

rank 0 独占终端与 TensorBoard/W&B logger；其他 learner 不刷新终端，也不创建独立
tfevents。rank 0 在每个 learner iteration 汇总跨 rank 标量：loss、reward 和 timing
取均值，计数与并行吞吐求和。`Perf/total_fps` / 终端 `Steps/s` 表示所有
collector 的聚合 env-step 吞吐；终端 `Rows/s` 表示所有 learner 的聚合 replay 行
吞吐（仅终端显示，不持久化；可由 run 配置与 `Perf/iteration_time` 推导）。checkpoint 同样由 rank 0 独占：每个保存间隔和
训练结束只在 canonical run 目录写一份模型；其他 rank 复用该路径完成进程协调，不创建
rank 子目录或任何日志文件。
自动生成的多卡 run 目录以 `_gpuxN` 结尾（例如 `_gpux2`）；单卡目录和显式
`training.log_dir` 保持原样。

collector 的 CPU 亲和按 rank 自动均分（`cpu_count // world_size` 一段），可用
`training.dp_collector_cpu_ids` 显式指定。该核区经 `EnvCfg.cpu_ids` 生效：除
MuJoCo worker 线程逐核绑定外，collector 进程及其已启动的本地线程也被限制在同一
核区内，避免跨 rank 抢占。

当前限制：

- 多卡 actor/critic optimizer CUDA Graph 支持捕获 NCCL gradient all-reduce。首次 capture
  前，DP owner 会先执行一次 eager all-reduce 并同步设备，完成 NCCL collective lazy
  initialization；flat-gradient buffer 在 graph 生命周期内保持固定地址。runtime manifest
  记录 `dp_sync.cuda_graph_optimizer_capture=enabled_after_collective_warmup` 和
  `cuda_graph_collective_warmup=true`。销毁 process group 前会先释放 graph 中的 NCCL 节点。
  Issue #978 的验证环境为 PyTorch 2.7.0+cu128、CUDA 12.8、NCCL 2.26.2、2 × RTX
  6000D；TCP loopback 下同步 all-reduce 和 `async_op=True` + `Work.wait()` 均可 capture/replay，
  默认 stream 与 side stream 均通过；跳过 warmup 时首个 all-reduce 会在 capture 中报
  `operation not permitted when stream is capturing`。有限超时的最小复现见
  `scripts/benchmark/rl/reproduce_nccl_cuda_graph_capture.py`。
- MuJoCo 有已提交的多卡 scaling benchmark；mjwarp 的 per-rank device placement 有
  `tests/base/backend/test_process_device.py` 与 off-policy runner/worker 单测覆盖，但仓库中
  尚无 mjwarp 多卡吞吐或收敛 benchmark。
- IsaacGym、IsaacSim 和 Genesis 的 collector 环境也会收到 rank 对应的仿真设备。off-policy
  使用父进程可见的 CUDA 索引；Genesis 在初始化 `gs.init` 前绑定进程级 session 设备。
- 仅单节点：rank 之间通过 run 目录里的 FileStore rendezvous，NCCL 走 TCP
  loopback（默认 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1`，环境变量显式设置
  时优先）——部分机型（如 RTX 6000D）的 NCCL P2P/SHM peer transport 不可靠，
  TCP loopback 是唯一稳定传输。

collector `Steps/s` 与 learner `Samples/s` 各自相对单卡的 scaling 基准见
`scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py`（issue #968，真实运行不进 CI）。

## 配置参考

默认值以 `src/unilab/structured_configs.py` 中的类型化配置（`SACConfig`）为准；
`src/unilab/conf/sac/config.yaml` 中的取值与之相同，除非另行注明。所有字段都可以在
命令行上 override：

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.gamma=0.99 \
  algo.algo_params.alpha_init=0.1
```

### 训练效率

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 4096 | 并行采集环境数 |
| `algo.batch_size` | int | 8192 | 每次 learner 更新的 batch |
| `algo.replay_buffer_n` | int | 512 | replay 容量（以整轮 env 向量为单位；行数 = n × num_envs） |
| `algo.updates_per_step` | int | 4 | 每个 env step 的 learner 更新数 |
| `algo.learning_starts` | int | 1 | learner 开始更新前需采集的整轮 env 向量数 |
| `algo.policy_frequency` | int | 4 | 每 N 次 learner 更新做一次 actor/target 更新 |
| `algo.max_iterations` | int | 500 | 训练迭代数 |
| `algo.save_interval` | int | 500 | checkpoint 保存间隔（迭代数） |
| `algo.algo_params.use_compile` | bool | true | learner 更新的 torch.compile |
| `algo.seed` | int | 1 | 随机种子 |
| `training.use_amp` | bool | true | learner 更新路径上的 bf16 autocast |
| `training.log_interval` | int | 1 | TensorBoard/W&B 标量上报间隔（迭代数） |

### 策略性能

| Override key | 类型 | 默认值 | 实际影响 |
| --- | --- | --- | --- |
| `algo.gamma` | float | 0.97 | 折扣因子 |
| `algo.tau` | float | 0.125 | target critic 的 EMA 系数 |
| `algo.actor_lr` | float | 3e-4 | actor 的 Adam 学习率 |
| `algo.critic_lr` | float | 3e-4 | critic 的 Adam 学习率 |
| `algo.actor_hidden_dim` | int | 512 | actor 网络宽度 |
| `algo.critic_hidden_dim` | int | 768 | critic 网络宽度 |
| `algo.num_atoms` | int | 101 | 分布式（C51）critic 的 atom 数 |
| `algo.obs_normalization` | bool | false | 归一化观测 |
| `algo.use_layer_norm` | bool | true | 网络中使用 LayerNorm |
| `algo.algo_params.alpha_lr` | float | 3e-4 | 熵温度（alpha）的学习率 |
| `algo.algo_params.alpha_init` | float | 0.01 | 初始熵温度 |
| `algo.algo_params.target_entropy_ratio` | float | 0.0 | 目标熵 = −action_dim × 该比例（0.0 → 目标为 0） |
| `algo.algo_params.max_grad_norm` | float | 0.0 | 梯度范数裁剪；0 表示不裁剪 |
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
与 `params`（示例：`src/unilab/conf/sac/task/g1_walk_flat/base.yaml`）。调整某个
项可以编辑该 owner YAML，也可以在命令行覆盖其权重：

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  reward.tracking_lin_vel.weight=2.5
```

各项名称与默认权重因任务而异；请查看所训练任务 owner YAML 的 `reward:` 段
（`src/unilab/conf/sac/task/<task>/base.yaml`）。
