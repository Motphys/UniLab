# 后端支持矩阵

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Backend 选择规则

- 当前 tensor-only Manager runtime 只支持 `mujoco`、`mjwarp`、`genesis`、
  `newton`、`motrix`。
- 默认后端是 `mujoco`
- `--sim mjwarp` 使用前需安装 `mjwarp` extra；已完成验证的组合以下方生成矩阵为准，其他入口也按矩阵查证
- `--sim genesis` 使用前需安装 `genesis` extra；真实 CUDA 依赖和平台约束见生成矩阵
- `--sim newton` 使用前需安装 `newton` extra；真实 CUDA 依赖和平台约束见生成矩阵
- `--sim motrix` 使用前需安装 `motrix` extra；CPU-authoritative packed
  HOST_BRIDGE profile 见生成矩阵
- `--algo`、`--task`、`--sim` 共同选择 owner YAML
- 不要把 `training.sim_backend` 当独立 backend switch
- `drake`、`isaacgym`、`isaacsim`、`superdex` 暂时不在本 runtime 支持范围内。adapter 保留在 UniSim，但不构成 UniLab 生产支持声明；重新启用需要 capability、parity 和支持矩阵证据（#1811）

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
| `mujoco` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | unknown | unknown | 不支持 | 支持 |
| `motrix` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 支持 |
| `mjwarp` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `newton` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `genesis` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix | mjwarp | Newton | Genesis |
|------------|------------|---|---|---|---|---|
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Tested | Registered | Configured |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | - | - | - |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | - | - | - |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | - | - | - | - |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | - | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | - | - | - | - |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | - | - | - | - |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | - | - | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | - | - | - |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | - | - | - |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | - | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | Configured | Registered | Configured |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Configured | - | - |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | - | - | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Configured | Registered | Registered |
| FlashSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | Configured | Configured | Configured |
| WarpSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Tested | Registered | Registered |
| WarpSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | Tested | Registered | Registered |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner backend identity: `training.sim_backend` in `src/unilab/conf/{ppo,appo,sac,flashsac,warpsac}/task/**`.
- Platform/capability source: `unisim.support.get_tensor_platform_profiles()`.
- Unsupported platform/device requests are guarded before backend construction in `src/unilab/base/backend_factory.py`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
<!-- END GENERATED SUPPORT MATRIX -->

## 平台故障排查

CUDA-only tensor 后端在 CPU、MPS、macOS、ROCm/HIP、CUDA 不可用、CUDA 序号
非法以及设备越界时，都会在后端构造前失败；不会回退到 NumPy 或 host bridge。

请在同一个进程环境中检查 Torch runtime 与可见序号：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())"
```

CUDA-only 矩阵要求 `torch.version.hip` 为 `None`。若 CUDA 不可用，先检查
NVIDIA 驱动与容器运行时不匹配，再调整任务配置。设置了
`CUDA_VISIBLE_DEVICES` 时，后端序号指向重映射后的命名空间，而不是宿主机
全局物理索引。

在 macOS 与 ROCm 上请使用 CPU-authoritative host-bridge 后端。host-bridge
后端使用 CUDA Torch buffer 不代表 CUDA physics，也不代表 device-resident
lifecycle。
