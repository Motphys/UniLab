# Tensor Runtime 生产指南

本页是 scoped 单 GPU G1 Motion Tracking / FlashSAC / MJWarp tensor runtime
的复现与运维入口，覆盖 canonical owner-routed 长 soak、runtime artifact 与
故障排查。设置契约见 {doc}`4-tensor_runtime`；所有 backend、task、入口与
平台支持声明的唯一权威是生成的
{doc}`../../5-reference/5-support_matrix`。

生产路径刻意保持很窄：

- Linux 与一块 NVIDIA CUDA GPU；
- FlashSAC；
- `g1_motion_tracking`；
- 通过统一 task-owner 路由选择的 `mjwarp` owner；
- Manager runtime 默认 tensor-native。

它不是多 GPU 指南，不发布软件包，也不能推广为所有 MJWarp task 的通用
声明。macOS 与 ROCm/HIP 没有该 device-resident lifecycle 的 fallback；
非法请求会在 backend 构造前失败。

## Owner 选择与有效组合

`--algo`、`--task` 与 `--sim` 必须一起使用。它们组合
`src/unilab/conf/<algo>/task/<task>/<sim>.yaml` 下的 owner YAML。真正声明
`training.sim_backend` 的是该 owner，而不是 CLI 拼写本身；tensor 执行是 Manager 不变量。

canonical 组合使用：

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp
```

MJWarp owner 继承 FlashSAC MuJoCo owner，并解析为：

| 有效设置 | 值 |
| --- | --- |
| Task 语义名 | `G1MotionTracking` |
| Backend 身份 | `mjwarp` |
| Tensor runtime | Manager 生命周期不变量 |
| Inference slot | `training.inference_slot_capacity=1` |
| Replay-ingress depth | `training.replay_ingress_depth=2` |
| Replay-ingress slot rows | `null`，解析为 `algo.num_envs` |
| Collector metric interval | `training.collector_metrics_interval=100` |
| CUDA 进程共享 | `training.cuda_process_sharing=null` |
| Scene | `src/unilab/assets/robots/g1/scene_flat.xml` |
| Motion | `motions/g1/dance1_subject2_part.npz` |

MuJoCo owner 使用同样的 2048 环境训练预算，但每步发布 collector metric。
MJWarp task 也是 DR-free lifecycle baseline：其 selected-row tensor reset
不协商 reset randomization 或 interval wrench。

在当前生成证据等级中，FlashSAC / `g1_motion_tracking` / MJWarp 是
`Configured`。本地 soak 成功不会自动把它提升为 `Benchmarked` 或
`Recommended`；支持等级提升必须通过 support-matrix generator 及其
validation inventory 单独决策。

## 可复现 Runtime

gated profile 通过已提交的 registry lock 文件解析 UniSim 与
MJBatch（RL 栈已内置于 `unilab.rl`）：

```bash
UV_FROZEN=1 uv sync --extra mujoco --extra mjwarp
```

`make setup` 适合日常仓库开发，但不安装 `mjwarp` extra。上面的命令与 gated
CUDA CI profile 一致。虽然包支持更宽的 Python 范围，CI 以 Python 3.11 作为
参考版本。

只可见一块 GPU 的主机无需设置 `CUDA_VISIBLE_DEVICES`。自动单设备拓扑会把
learner、collector、MJWarp 物理后端、inference ring 与 replay ingress 绑定到
同一个当前 CUDA 设备。只有在多 GPU 主机上选择一块物理 GPU，或需要保留
launcher/debug/MPS 掩码时才设置该变量：

```bash
export CUDA_VISIBLE_DEVICES=<single-host-cuda-ordinal>
```

trainer 进程内所有 CUDA ordinal 都相对于该 mask。不要添加更多设备：M11 仅
覆盖单 GPU。

## CUDA MPS 执行共享

`training.cuda_process_sharing` 是针对既有 rank-local 拓扑的显式
execution-sharing 模式。它不是 backend 开关，也不替代 `--sim mjwarp` 或 owner
YAML 选择。

默认 `null` 表示 learner 与 collector 保持独立 CUDA context。初始支持拓扑为
Linux、NVIDIA CUDA、单物理 GPU、MJWarp、SAC/FlashSAC 且 `world_size=1`。使用以下
设置请求共享 GPU 执行：

```bash
training.cuda_process_sharing=mps
```

### 受管理的 daemon 生命周期

UniLab 提供显式的用户自有生命周期命令，但训练路径仍不会隐式启动或停止 host
服务：

单 GPU 主机无需传任何参数，默认值已经完整：

```bash
uv run uni-cumps start
uv run uni-cumps doctor
uv run uni-cumps stop
```

`start` 后，设置 `training.cuda_process_sharing=mps` 的 trainer 会自动选择
覆盖该 GPU 的唯一 live UniLab-recorded daemon。Shell 集成变为可选：

```bash
uv run uni-cumps start
uv run --extra mjwarp train --algo flashsac \
  --task g1_motion_tracking --sim mjwarp \
  training.cuda_process_sharing=mps
```

调用方显式提供的 `CUDA_MPS_PIPE_DIRECTORY` 仍然优先。若没有匹配的 live
daemon，训练会在 environment/learner/collector 构造前 fail closed，并指向
`uni-cumps start`。

默认参数：

| 设置 | 默认值 |
| --- | --- |
| GPU | 唯一可见的物理 GPU，并解析为 canonical UUID |
| daemon 名称 | `gpu-<full-uuid>` |
| pipe 目录 | `/tmp/uni-cumps/<name>/pipe` (mode `0700`) |
| log 目录 | `/tmp/uni-cumps/<name>/log` (mode `0700`) |
| `env`/`stop` 目标 | 唯一 live 的 UniLab-recorded daemon |

如果可见 GPU 多于一张，`start` 与 `doctor` 必须显式传
`--gpus <index-or-uuid>`。需要稳定 label 或服务自有路径的部署仍可显式覆盖：

```bash
uv run uni-cumps start \
  --gpus <gpu-index-or-uuid> \
  --name <daemon-name> \
  --pipe-dir /absolute/path/mps/pipe \
  --log-dir /absolute/path/mps/log
eval "$(uv run uni-cumps env --name <daemon-name>)"
uv run uni-cumps stop --name <daemon-name>
```

只读 status 无参数即可使用：

```bash
uv run uni-cumps status
```

`status` 与 `doctor` 是只读诊断。`start` 创建 user-owned daemon，并在当前用户的
UniLab cache 中写入 record。`stop` 只接受 record 中的 ownership evidence：UID、
host、control PID 与 Linux process start-time；它绝不会向 unmanaged pipe 发送
`quit`。`env` 输出给 launcher 或显式 shell 集成使用的环境变量，不修改 UniLab
父进程环境。

daemon 不是某个训练 run 的独占资源，可以服务多个 client。stale record 会在复用前
被 quarantine；stop 后保留日志。

### 拓扑模式与当前限制

命令会报告三个 topology-mode 名称，使未来 launch contract 保持加法扩展：

| 模式 | 含义 | 初始状态 |
| --- | --- | --- |
| `single_gpu` | 一个任务使用一张物理 GPU | 已实现 |
| `single_task_multi_gpu` | 一个任务每张 GPU 一个 rank | 可解析，fail-closed |
| `task_per_gpu` | 多个独立任务按 GPU 打包 | 可解析，fail-closed |

selector 会解析成 canonical 物理 GPU UUID。MIG UUID 显式拒绝。逗号分隔的多 GPU
请求会 fail closed，并指向 multi-GPU gate，而不是静默选择 DP 或 task packing。
`all` 可用于只读诊断，但当前版本不能用它启动 daemon。

训练仍在 environment probe、learner construction 与 collector spawn 之前验证；
没有静默 multi-context fallback。错误会指出第一个未满足条件，并指向 CLI 生命周期
命令。

`run_config.json` 记录配置值。有效 run 的 `run_summary.json` 内嵌
`runtime_manifest.cuda_process_sharing`，包含 configured/effective 模式、learner
与 collector device、物理 UUID 证据、control pipe、server PID 与 validation 状态。
未配置（`null`）的 run 在 CUDA rank 上仍会记录环境级 host 状态——
`CUDA_MPS_PIPE_DIRECTORY` 原始值、生效的 pipe 目录与 control pipe 类型、应答的
server PID，以及 best-effort 的 learner/collector GPU UUID——当环境级 daemon 在
生效 control pipe 上应答时 `ambient_server=true`，使透明走 MPS 的 run 可以从自身
manifest 自证。ambient 观测永不 fail closed；无法解析的事实记录为 `null`。
该 section 是 runtime-manifest v1 的 producer diagnostic，不是稳定标量契约。

MPS 只改变 GPU execution sharing，不改变 `env_steps_per_sync`、
learner/collector placement 或训练语义。多 GPU DP 在独立的 gate 与 daemon 拓扑
决策完成前保持不支持。CUDA MPS 仍依赖 host 与部署方式：在共享容器、多用户主机或
受限 runner 中 control 可能不可用，显式请求会 fail closed 而不是静默降级。

无法使用 UniLab CLI 的受限部署仍可以用 `nvidia-cuda-mps-control` 手动运行既有
daemon；训练验证不关心该 daemon 由哪个显式部署工具启动。

检查实际执行 benchmark 的 runtime：

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())"
```

`torch.version.hip` 必须为 `None`，CUDA 必须可用，且可见设备数必须为 1。

## 后端 CUDA Graph 诊断

训练 artifact 会记录物理 backend 的 tensor runtime 诊断。
`TorchEnv.get_backend_runtime_diagnostics()`（`src/unilab/base/torch_env.py`）
是采集边界：一个 public cold-path 方法，把 backend 的
`SimBackend.get_tensor_runtime_diagnostics()`（unisim 公开契约）序列化为
纯 JSON。off-policy collector（`unilab.rl.runtime.offpolicy`）在启动时调用它，并把 payload 发布到
`runtime_manifest.backend_diagnostics`；PPO 没有 runtime manifest，因此
PPO 训练脚本（`src/unilab/scripts/train_rsl_rl.py`）把同样的 payload 作为
顶层 `backend_diagnostics` 字段写入 `run_summary.json`。

payload 形状：

```json
"backend_diagnostics": {
  "available": true,
  "backend_type": "mjwarp",
  "diagnostics": {
    "cuda_graph": {"requested": true, "enabled": false, "disable_reason": "..."},
    "selected_reset_sensor_refresh": {"requested": true, "enabled": true, "disable_reason": null},
    "device_reset_randomization": {"requested": true, "enabled": true, "disable_reason": null}
  }
}
```

- `requested`：该优化已被请求（MJWarp 总是请求 CUDA graph）。
- `enabled`：该优化实际生效（例如 fixed-address CUDA graph replay）。
- `disable_reason`：backend 自有的 fallback 原因；`enabled` 为 false 时
  必为非空字符串，`enabled` 为 true 时必为 `null`。

当 env 或 backend 不提供诊断时，payload 是显式的 unavailable 标记，绝不
是裸 `null`：

```json
"backend_diagnostics": {"available": false, "backend_type": "...", "unavailable_reason": "..."}
```

与 `runtime_manifest.cuda_process_sharing` 一样，该 section 是
runtime-manifest v1 的 producer diagnostic，不是稳定标量契约；runtime
manifest schema（`unilab.rl.logging.runtime_manifest_schema`）
在该 section 存在时会 fail-closed 校验。这些诊断
仅用于 observability：不改变 CUDA graph 启用策略，也不是验收 gate。它们是
collector/env 启动时采集的 cold-path 快照，描述初始化状态，不描述 per-step
行为。

| 现象 | 首要处理 |
| --- | --- |
| 某项优化 `enabled: false` | 读取 `disable_reason` 获取 backend 自有的 fallback 原因（graph capture 失败、拓扑不支持等）；run 仍然有效，只是按 eager 方式 step。 |
| `available: false` | 查看 `unavailable_reason`：要么 backend 未声明 tensor runtime 诊断（CPU MuJoCo 等非 tensor backend 上属预期），要么诊断 probe 抛出异常（包含异常类型）。 |

已知限制：unisim 尚未暴露 CUDA graph 版本或 eager-fallback 事件计数。
serializer 会自动透传未来的 dataclass 字段，因此该 section 会随 unisim 的
扩展而增长。

## 资产

G1 mesh 与纹理来自 asset hub。离线 benchmark 前先预取：

```bash
uv run --no-sync unilab-pull-assets --robot g1
```

motion clip 同样不在 Git 中。它会在首次使用时惰性下载，并必须存在于：

```text
src/unilab/assets/motions/g1/dance1_subject2_part.npz
```

完全离线运行前，先物化所需 motion 文件，再设置 `HF_HUB_OFFLINE=1`。首次
物化完成前不要开启 offline 模式。

## Canonical 长 Soak

长 soak 是 canonical production benchmark。它启动真实 owner-routed
FlashSAC trainer，并监控 schema、进度、CUDA 状态、进程、file descriptor、
shared memory、inference flight、replay ingress 与 shutdown。

为 run 目录和 JSON artifact 选择全新的绝对路径。不要复用脏 run 目录：
monitor 使用该目录中的文件变化作为 liveness 信号。

```bash
uv run --no-sync scripts/benchmark/torch_env/g1_flashsac_soak.py \
  --num-envs 1024 \
  --iterations 65000 \
  --min-duration-seconds 1800 \
  --sample-interval-seconds 5 \
  --startup-timeout-seconds 900 \
  --stale-progress-seconds 300 \
  --post-shutdown-grace-seconds 10 \
  --extra-override algo.save_interval=10000 \
  --log-dir /absolute/path/to/g1-flashsac-mjwarp-run \
  --output /absolute/path/to/g1-flashsac-mjwarp-soak.json
```

wrapper 拥有并拒绝覆盖 `algo`、`task`、`training.sim_backend`、play 模式、
log 目录、环境数与迭代数。它为 `flashsac` / `g1_motion_tracking` /
`mjwarp` 构造底层 owner 路由；用户不应把
`training.sim_backend=mjwarp` 当作独立开关传入。

M11 验收预算刻意不同于 owner 生产默认值：

| 预算 | 环境数 | 迭代数 | 用途 |
| --- | ---: | ---: | --- |
| MJWarp owner 默认 | 2048 | 25000 | Owner 训练预算 |
| M11 长 soak | 1024 | 65000 | lifecycle、资源、replay 与 shutdown 验收 |

wrapper 还设置 `training.no_play=true`，因此通过的 soak 不会生成 playback
视频或 playback 驱动的 ONNX。

## 必需 Artifact

run 目录必须包含：

- `run_config.json`；
- `run_summary.json`；
- TensorBoard event 文件；
- 按所选 `algo.save_interval` 生成的 checkpoint；
- soak monitor 消费的进度文件。

`run_summary.json` 必须携带 metric schema `1` 与内嵌 runtime-manifest
schema `1`。manifest 记录有效 runtime limits、发生预算决策时的 inference /
tensor memory budget、collector/backend device、inference-flight 状态、
replay-ingress 状态、shutdown diagnostics，以及 backend tensor runtime 诊断
（`runtime_manifest.backend_diagnostics`；PPO run 以顶层
`backend_diagnostics` 携带同样的 payload）。字段与兼容契约见
{doc}`3-logging`。

soak JSON 是自包含 monitor evidence，artifact schema 为 `0.3.0`。当前 head
通过必须满足：

| 证据 | 通过值 |
| --- | --- |
| Monitor 状态 | `passed` |
| Trainer 返回码 | `0` |
| Run 状态 | `completed` |
| Failure reason 与 classification | 均为 `null` |
| 最终 inference queue depth 与 publication lag | 均为 `0` |
| Replay 最终 occupancy | `0` |
| Replay dropped batches 与 early returns | 均为 `0` |
| Replay release sequence | 等于 published sequence |
| Published sequence | `total_env_steps / ingress_slot_rows` |
| Shutdown 分类 | `normal_completion` |
| Shutdown cleanup 错误 | `[]` |
| Post-shutdown residual processes | `[]` |

`resource_summary` 另外记录 process、RSS、GPU memory、eventfd、
shared-memory FD、replay high-water occupancy、inference event count 与最大
in-flight request 的峰值。出现峰值是正常的；需要关注的是 post-startup
样本中的无界或持续增长，而不是非零最大值本身。

artifact 的 `context` 记录 UniLab commit 与 dirty 状态、GPU UUID/名称/驱动、
Python/Torch 元数据与 `CUDA_VISIBLE_DEVICES`。JSON 与 run 目录需要一起
保留。仓库有意不跟踪这些机器生成 artifact。

运行后提取最后 20 个 schema-v1 TensorBoard 窗口：

```bash
uv run --no-sync scripts/benchmark/rl/extract_offpolicy_metrics.py \
  /absolute/path/to/g1-flashsac-mjwarp-run \
  --last 20 \
  --json
```

helper 遇到缺失或不支持的 metric schema 会 fail closed。历史 unversioned
event 文件必须显式使用 `--legacy-unversioned`，永远不会作为隐式 fallback。

## 历史 Soak 证据

M11 lifecycle review 在本地 RTX 5090 上获得通过的 v6 长 soak：65,000 迭代、
66,660,352 环境 step、1987.55 秒、无 dropped replay batch、最终 replay
occupancy 为 0、最终 inference queue/lag 为 0、normal shutdown 且无 residual
process。资源采样显示 process、RSS、GPU memory、eventfd 与 shared-memory FD
均有界。

该 v6 artifact 是历史证据，不是当前 head 的 benchmark 声明。它使用 soak
artifact schema `0.2.0`，且 UniLab、unilab-rl、UniSim commit 早于当前
schema-v1 consumer enforcement。需要当前 head 证据时，请用上面的命令在确切
checkout 上重新执行。

## 可选 Phase-Local Probe

`g1_flashsac_backend.py` 是诊断 probe，不是 canonical 端到端 benchmark。它
刻意排除 inference IPC、replay ingestion、learner update 与生产
Manager-Based dispatch。

```bash
uv run --no-sync scripts/benchmark/torch_env/g1_flashsac_backend.py \
  --backends mjwarp,mujoco \
  --num-envs 2048 \
  --warmup 20 \
  --iters 100 \
  --acceptance \
  --output scripts/benchmark/outputs/g1-flashsac-backend/current-host.json
```

acceptance 模式要求可查询 `nvidia-smi`，且每个隔离 backend 进程前后 GPU 均
空闲。它还会拒绝 Isaac worker profiler 环境变量。

JSON schema 为 `0.3.0`。跨 backend 比较使用
`throughput_env_control_steps_per_s` 与同步后的
`phases.iteration_ms`。单个 phase 边界可能按 stream 排序，因此不要把 phase
相加成新的墙钟总量，也不要把一个 backend 的 `update_state_ms` 直接与另一个
backend family 的同名 phase 比较。MuJoCo 结果还包含 packed H2D/D2H 清单与
transfer counter。

## 故障排查

| 现象 | 首要处理 |
| --- | --- |
| MJWarp import 失败 | 重新执行显式 `uv sync --extra mujoco --extra mjwarp`；仅 `make setup` 不足以覆盖该 profile。 |
| CUDA 不可用 | 检查 driver/container runtime 与 Torch sanity 命令；不要修改 task owner 强制 fallback。 |
| 物理 GPU 错误 | 在启动 `uv run` 的 shell 中设置 `CUDA_VISIBLE_DEVICES`；backend ordinal `0` 相对该 mask。 |
| CUDA IPC 或 worker 不匹配 | 保持 learner、collector 与 payload 在同一物理 GPU。外部 Isaac worker 继承可见性，但不继承 host Python path。 |
| 资产启动失败 | 预取 G1 资产，确认 motion NPZ 存在；首次下载完成前移除 `HF_HUB_OFFLINE=1`。 |
| startup timeout | 查看 `soak-console.log` 与第一条 monitor sample；区分资产下载、依赖安装、backend startup 与进程失败。 |
| stale progress | 打开最新 run 文件与 console log；超过 `stale-progress-seconds` 没有文件变化会显式失败，而不是无限等待。 |
| replay backpressure 或 drop | 读取 occupancy、high-water、wait、early return 与 dropped batch；未做 workload benchmark 前不要缩小 ingress slot rows。 |
| 最终 replay occupancy 非零 | 保留 artifact 与 console log；即使 trainer 看似完成，这也是 lifecycle/finalize 失败。 |
| eventfd/shared-memory 增长 | 对比 post-startup 样本；有界峰值正常，持续增长是泄漏信号。 |
| abnormal shutdown | 用 `failure_classification` 与 `shutdown.classification` 区分 learner、collector、backend worker、stale tick、cancellation 或 unknown failure。 |
| residual process | 保留记录已知 PID 与 start tick 的 artifact；在进程被解释并消失前不要复用 GPU。 |
| unsupported capability | 阅读精确 fail-closed 错误与生成的 support matrix；CUDA-only lifecycle 没有 hidden NumPy 或 host-bridge fallback。 |

完整 schema 定义与 metric 判读见 {doc}`3-logging` 与
{doc}`4-tensor_runtime`。

## Tensor Runtime 破坏性迁移

`env.tensor_runtime` 与 `env.tensor_runtime_device` 已移除，且没有兼容别名。
Manager runtime 构造上就是 tensor-native：设备位置由 backend 声明的数据面与
rank 进程设备推导。此前设置这两个字段的 owner YAML 和 runner override 必须
删除它们。不支持的 tensor lifecycle 会在 binding 阶段失败，不会回退到 NumPy
wire。
