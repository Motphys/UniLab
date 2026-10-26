# 后端支持矩阵

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Backend 选择规则

- 当前 tensor-only Manager runtime 支持 `mujoco`、`mjwarp`、`genesis`、
  `newton`、`motrix`、scoped Drake owner，以及下方显式 external-worker
  Isaac owner。
- 默认后端是 `mujoco`
- `--sim mjwarp` 使用前需安装 `mjwarp` extra；已完成验证的组合以下方生成矩阵为准，其他入口也按矩阵查证
- `--sim genesis` 使用前需安装 `genesis` extra；真实 CUDA 依赖和平台约束见生成矩阵
- `--sim newton` 使用前需安装 `newton` extra；真实 CUDA 依赖和平台约束见生成矩阵
- `--sim motrix` 使用前需安装 `motrix` extra；CPU-authoritative packed
  HOST_BRIDGE profile 见生成矩阵
- `--sim drake` 需要本地 Drake extra 与 DrakeUni batch extension；当前 scoped
  支持范围仅为 PPO `go2_joystick_flat`
- `--sim isaacgym` 与 `--sim isaacsim` 需要各自隔离的外部 Python worker 以及
  同 GPU CUDA IPC；安装方式见对应后端页面。
- `--algo`、`--task`、`--sim` 共同选择 owner YAML
- 不要把 `training.sim_backend` 当独立 backend switch
- Isaac 支持等级继续受 task/evidence 约束；`superdex` 仍限于已配置的
  research owner。

## Playback Differences

- `mujoco`: `--render-mode auto` 会导出 `play_video.mp4`；`--render-mode viser`
  通过基于浏览器的 viser viewer 展示回放
- `mjwarp`: 默认仅支持显式、有限步数的 `record`，通过 task owner 的 MuJoCo visual model 离线录制；`--render-mode interactive` 路由到 MuJoCo 交互 viewer（mjwarp 跑物理、MuJoCo 渲染 env[0]，强制单 env）；`--render-mode viser` 路由到浏览器 viser viewer（按 env 使用 MuJoCo playback model）；不支持 `auto` 或 native renderer
- `genesis`: 当前不支持 `viser`，也不支持这里的 MuJoCo 交互/离线 playback 契约
- `newton`: canonical owner 使用 `record` 原生 Newton 渲染；安装不完整时回退
  MuJoCo snapshot
- `--render-mode record`: MuJoCo 与 mjwarp 只录制视频
- `--render-mode none`: 不回放

## Support Matrix

下面的矩阵由 registry、owner YAML backend identity、测试/验证清单和 UniSim 平台
profile 自动汇总；不要手工编辑表格内容。需要刷新时运行：

```bash
uv run scripts/generate_support_matrix.py --write
```

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Evidence Grades

| 等级 | 仓库事实来源 |
|------|--------------|
| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |
| `Configured` | owner YAML 的 `training.sim_backend` 指向该 backend。 |
| `Tested` | 自动化覆盖或显式 maintainer 完整训练验证；不等同于默认推荐路径。 |
| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |
| `Recommended` | 仓库中存在显式 recommendation 元数据。 |

`Tested` 只描述仓库证据，不表示同名 MuJoCo owner 的全部 DR、渲染或 production 能力。当前没有已提交 benchmark/recommendation 元数据，因此不会自动提升到 `Benchmarked` 或 `Recommended`。

### Tensor Backend Platform Matrix

该表来自 UniSim SDK-free 公开静态能力清单；它表示 reviewed tensor lifecycle 边界，不表示可选 SDK 已安装，也不把所有 task owner 自动提升为可用组合。

| Backend | Execution / process / data plane | Torch devices | CUDA runtime | Linux+CUDA | macOS | ROCm | Worker | Reset randomization | Fixed variants | Host callbacks | Packed bridge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mujoco` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge with explicit ROCm Torch buffers (`cuda`) | In-process; no external Python worker | unknown | unknown | 不支持 | 支持 |
| `motrix` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge with explicit ROCm Torch buffers (`cuda`) | In-process; no external Python worker | unknown | 不支持 | 不支持 | 支持 |
| `drake` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge with explicit ROCm Torch buffers (`cuda`) | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 支持 |
| `mjwarp` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `newton` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 支持 | 不支持 | 不支持 | 不支持 |
| `superdex` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge with explicit ROCm Torch buffers (`cuda`) | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 支持 |
| `genesis` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported UniSim tensor profile; Genesis engine natively supports Metal, with no CPU or substitute-backend fallback | Supported: Linux ROCm with HIP Torch and exact gs.amdgpu (cuda); no CPU or substitute-backend fallback | In-process; no external Python worker | 支持 | 不支持 | 不支持 | 不支持 |
| `isaacgym` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.8 worker; host Python paths are not inherited | 支持 | 不支持 | 不支持 | 不支持 |
| `isaacsim` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.11 worker; host Python paths are not inherited | 支持 | 不支持 | 不支持 | 不支持 |

配对契约（issue #2483，ADR-0019）：在非 ROCm 的 Torch 构建上，CUDA learner 下 `training.collector_tensor_device=cpu` 会 fail closed——上表 HOST_BRIDGE 的 `CPU` Torch 设备能力不等于 CUDA learner 可用该配对。CPU 载体对 CPU / torch.mps learner 与 ROCm/HIP 构建仍然合法；归因基线通过 `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1` 显式 opt-in。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix | Drake | mjwarp | Newton | SuperDex | Genesis | IsaacGym | IsaacSim |
|------------|------------|---|---|---|---|---|---|---|---|---|
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Configured | - | - | Configured | - | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Registered | - | Configured | Registered | Registered |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Registered | Registered | - | Registered | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Registered | - | - | Registered | - | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | - | Registered | Registered | - | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Registered | Registered | - | Registered | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Tested | - | Tested | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Registered | - | Configured | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Registered | Registered | - | - | Registered | - | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Configured | Registered | - | Registered | Registered | Registered |
| FlashSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Configured | - | Configured | - | - |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner backend identity: `training.sim_backend` in `src/unilab/conf/{ppo,appo,sac,flashsac}/task/**`.
- Platform/capability source: `unisim.support.get_tensor_platform_profiles()`.
- Unsupported platform/device requests are guarded before backend construction in `src/unilab/base/backend_factory.py`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
<!-- END GENERATED SUPPORT MATRIX -->

## 平台故障排查

CUDA-only tensor 后端在 CPU、MPS、macOS、ROCm/HIP、CUDA 不可用、CUDA 序号
非法以及设备越界时，都会在后端构造前失败；Genesis 则接受其声明的
HIP/ROCm + `gs.amdgpu` profile。不支持的请求不会回退到 NumPy 或 host bridge。

请在同一个进程环境中检查 Torch runtime 与可见序号：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())"
```

CUDA-only 后端与 Genesis `gs.cuda` 配对要求 `torch.version.hip` 为 `None`；
Genesis `gs.amdgpu` 配对则要求 HIP build。若 GPU runtime 不可用，先检查
驱动与容器运行时不匹配，再调整任务配置。设置了 `CUDA_VISIBLE_DEVICES` 时，
后端序号指向重映射后的命名空间，而不是宿主机全局物理索引。

在 macOS 上请使用 CPU-authoritative host-bridge 后端。ROCm 支持 host bridge
与 Genesis exact-`gs.amdgpu` device-resident profile。对于 host bridge，只要
learner 设备为 CUDA——默认的 `training.collector_tensor_device=auto` 会跟随
learner 设备——或训练显式设置 `training.collector_tensor_device=cuda`，且所选
后端声明并接受被路由的 `cuda` `env.collector_tensor_device`，Manager/TorchEnv
就使用当前 GPU；CPU learner 保持 CPU，torch.mps learner 保留 owner/default
CPU 放置，直接构造环境仍为 CPU。host bridge 使用 GPU Torch buffer 不代表
GPU physics，也不代表 device-resident 后端生命周期。
