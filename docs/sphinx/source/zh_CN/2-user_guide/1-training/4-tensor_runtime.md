# Tensor runtime

SAC 与 FlashSAC 会在探测环境或构造 learner 之前解析所有面向进程
启动的 tensor-runtime 设置。owner YAML 中的值会被收敛为一个有边界的
`RuntimeSettings` 对象并传给 off-policy runner。因此，非法值、超出
边界的数据以及显存预算不可能满足的组合，都会在 collector 或 learner 进程
启动前失败。

该契约只覆盖当前 scoped 单 GPU off-policy runtime；它不是自动调参器，也不
提供多 GPU scaling。

## Owner 默认值与边界

| 设置 | 默认值 | 有效范围或约束 |
| --- | ---: | --- |
| `training.inference_slot_capacity` | `1` | 正整数，最大 `16` |
| `training.collector_metrics_interval` | `1` | 正整数，最大 `10000` |
| `training.replay_ingress_depth` | `2` | 正整数，最大 `16` |
| `training.replay_ingress_slot_rows` | `null` | `1` 到 `algo.num_envs`；`null` 表示 `algo.num_envs` |
| `training.cuda_process_sharing` | `null` | `null` 或单 rank MJWarp 显式 `mps` |
| Learner 每次同步的行数 | `algo.batch_size * algo.updates_per_step` | 受 CUDA 显存预算约束 |

除 `training.cuda_process_sharing` 外，所有值必须是精确的正整数。布尔值、字符串、
浮点数、零和负数都会 fail closed。`training.cuda_process_sharing` 只接受 `null`
或显式字符串 `mps`。G1 Motion Tracking / MJWarp owner 仅有意将
`collector_metrics_interval` 覆盖为 `100`；其他 tensor-runtime 默认值仍保持
上表所示。

## 设备语义与解析

设备相关设置按消费层分离：

| 配置键 | 生效层 | 合法值 |
| --- | --- | --- |
| `training.collector_tensor_device` | 训练装配层，决定 collector tensor 驻留（HOST_BRIDGE Manager 载体与派生的 off-policy inference ring） | `auto`、`cpu`、不带序号的 `cuda` |
| `env.collector_tensor_device` | HOST_BRIDGE 后端的 Manager 公共 tensor | `cpu`、`cuda`、`cuda:<index>` |
| `training.device` | learner 与 rank-local 进程选择 | 具体 Torch device 标签 |

训练键是唯一面向用户的 collector tensor 驻留请求。`auto` 跟随已解析的 learner 设备：
CUDA 路由 rank-local 带序号设备，CPU 路由 CPU，torch.mps（Metal
Performance Shaders）保留 owner/default CPU 放置。显式 `cpu`/`cuda` 优先于
`auto`。`cuda` 是具体设备家族，绝不映射到 MPS；无 CUDA 主机上的 CUDA 请求
会 fail closed。

载体路由由能力声明驱动：MuJoCo、Motrix、Drake 与 SuperDex 声明了
in-process packed `HOST_BRIDGE` transfer plan，支持 `cpu`/`cuda` 载体。
其他后端忽略 `auto`/`cpu`，但 `cuda` 请求会 fail closed。请求超出所选后端
声明的 `torch_devices` 时同样 fail closed。`run_config.json` 保留组合后的
配置，并在 `run.resolved_devices` 的 `collector_tensor` 条目中记录
"配置值 → 解析值"映射；启动日志打印同一映射。

off-policy inference transport 是纯派生事实，不再是配置键：
`collector_tensor_device=cpu` 推导 host ring（观测与动作走 CPU inference
ring；默认合法配对中 learner 是 CPU 或 Apple MPS，CUDA learner 的配对约束
见下文的配对契约），`cuda` 推导 CUDA ring。`cuda`
遵循 Torch device type 语义，覆盖 NVIDIA CUDA 与 AMD ROCm。它不会放宽独立
平台 gate：CUDA-only tensor backend 在 ROCm/HIP 上仍 fail closed；Genesis
仅通过独立审查的 HIP/ROCm + exact `gs.amdgpu` profile 支持 ROCm。
`training.cuda_process_sharing` 仍只开放给 NVIDIA 专属的 CUDA MPS daemon 与
native H2D 实现。

已移除键会带迁移错误 fail closed：用 `training.collector_tensor_device` 替代
`training.sim_carrier_device` 与 `training.inference_transport`，用
`env.collector_tensor_device` 替代 `env.sim_carrier_device` 与
`manager_torch_device`，并用已解析的 learner 设备替代
`env.tensor_runtime_device`。

迁移说明：默认 `auto` 下的 CUDA learner 会通过 packed host bridge 呈现 CUDA
tensor，并派生 CUDA inference ring。此时每个 collector 进程会在
learner GPU 上额外持有一个 CUDA context（拓扑见
{doc}`7-tensor_runtime_production`）。

配对契约（issue #2483，ADR-0019）：在非 ROCm 的 Torch 构建上，显式
`training.collector_tensor_device=cpu` 与 CUDA learner 的组合会 fail
closed——CPU 载体不再是 CUDA learner 的性能路径（#2479 重测确认其与 CUDA
载体 + CUDA MPS 吞吐持平）。CPU 载体仍然合法的配对是：CPU learner、
torch.mps learner（Apple Silicon 上唯一的 host bridge 拓扑）与 ROCm/HIP
构建（无 CUDA MPS 等价物，CPU 载体是唯一进程共享缓解）。归因基线与调试
run 通过 `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1` 显式 opt-in 保留旧拓扑。

“worker”一词同样按层区分：

| 术语 | 含义 |
| --- | --- |
| Collector worker | APPO 或 off-policy 中推进环境并发布 rollout 的进程 |
| Backend worker | 后端自有外部模拟器进程（例如 Isaac） |
| torchrun rank | PPO 多 GPU launcher 启动的分布式 learner rank |
| Backend process device | 进程全局加速器绑定，不是 Manager carrier 键 |

## Replay ingress 权衡

默认 `replay_ingress_slot_rows: null` 会解析为 `algo.num_envs`，并为每个
collector vector 发布一个 ingress chunk。这使 publication 与同步模式保持
可预测，是推荐的生产默认值。

将该字段设置为更小的值可以把一个 collector vector 切成连续 chunk，从而降低
常驻 ingress 内存。它只应用于显式的内存压力调参。chunk 数量增加意味着更多
publication、更多 producer backpressure 机会，并且在 CUDA 上每个发布的 chunk
都会引入一次 current-stream barrier。采用更小的值前，务必在目标 workload 上
重新 benchmark。

Ingress diagnostics 统计的是 publication chunk，而不是完整的逻辑 collector
vector。shutdown 时可能保留一个尚未完整发布 vector 的有效行前缀，但不会发布
非法 transition 行。replay ingress 计数器与 shutdown schema 见
{doc}`3-logging`。

## 随机数生成

Manager 环境持有 `unilab.managers.ManagerRng`，它是训练 runtime 的
唯一 RNG。该适配器包装一个显式 CPU/CUDA `torch.Generator`，校验种子和
未支持的采样语义，并始终在该 generator 的设备上返回 tensor。它不是 NumPy
bitstream 兼容层。训练入口 seed 契约只 seed 全局 Python 与 Torch RNG，不再
修改 legacy 全局 NumPy RNG。

当前 scoped G1 Motion Tracking / FlashSAC owner 使用 CUDA
`ManagerRng` 驱动观测噪声、adaptive/mixed motion sampling 与
selected-row reset。因此 A/B 评估比较同一整数种子下的可复现性和分布，而不是
逐位比较 NumPy 轨迹。确定性运行必须显式设置 `env.seed`；`algo.seed` 只 seed
训练进程的全局 Python/Torch RNG，不会替换 owner 本地 generator 的种子。

`ManagerRng.get_state()` 返回 generator 的 CPU state tensor，
`set_state()` 负责恢复。可复现性的权威载荷就是精确的 generator state；
恢复或审计 RNG 的 owner 记录该 state tensor 即可。runtime 不再维护逐
transition 的版本号簿记。

## Runtime evidence

Runtime manifest 记录审计 run 所需的有效证据：

- `runtime_limits` 包含已解析 tensor-runtime knob 的 configured、default、
  effective 与 maximum 值。
- `inference_memory_budget` 记录有边界的 CUDA inference-ring 预算。
- `tensor_memory_budget` 记录 CUDA inference、replay storage、replay ingress、
  learner batch 与 workspace 的组合预算计算。
- owner 显式请求 `mps` 时，`cuda_process_sharing` 记录已验证的
  execution-sharing 证据；默认 `null` 模式下该字段不存在。

预算决策在 spawn 前写入 manifest。如果不安全的组合被拒绝，错误会指出具体
设置，且相关进程不会被启动。

单 GPU G1 FlashSAC/MJWarp 生产工作流的 benchmark 复现、artifact 与故障排查见
{doc}`7-tensor_runtime_production`。
