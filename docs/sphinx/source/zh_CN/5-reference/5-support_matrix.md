# 后端支持矩阵

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Backend 选择规则

- 默认后端是 `mujoco`
- 切到 Motrix 用统一 CLI 的 `--sim motrix`
- `--sim mjwarp` 使用前需安装 `mjwarp` extra；已完成验证的组合以下方生成矩阵为准，其他入口也按矩阵查证
- `--algo`、`--task`、`--sim` 共同选择 owner YAML
- 不要把 `training.sim_backend` 当独立 backend switch

## Playback Differences

- `mujoco`: `--render-mode auto` 会导出 `play_video.mp4`；`--render-mode viser`
  通过基于浏览器的 viser viewer 展示回放
- `motrix`: `--render-mode auto` 会打开交互式 renderer 窗口，不录制视频，不受 `play_steps` 限制；`--render-mode viser` 路由到浏览器 viser viewer（物理快照驱动按 env 的 MuJoCo playback model）
- `mjwarp`: 默认仅支持显式、有限步数的 `record`，通过 task owner 的 MuJoCo visual model 离线录制；`--render-mode interactive` 路由到 MuJoCo 交互 viewer（mjwarp 跑物理、MuJoCo 渲染 env[0]，强制单 env）；`--render-mode viser` 路由到浏览器 viser viewer（按 env 使用 MuJoCo playback model）；不支持 `auto` 或 native renderer
- `isaacsim`: `auto` 在有 display 时选择 Kit viewer，否则选择 headless RGB camera；当前真实主机仍有 RTX renderer 初始化 blocker，支持等级保持 `Configured`
- `--render-mode record`: MuJoCo、mjwarp、Motrix 都只录制视频；IsaacSim 路由到离屏 RGB 协议，真实主机 playback 支持仍保持 `Configured`
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
| `drake` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 支持 |
| `mjwarp` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `newton` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `superdex` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 支持 |
| `genesis` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | 不支持 | 不支持 | 不支持 | 不支持 |
| `isaacgym` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.8 worker; host Python paths are not inherited | 不支持 | 不支持 | 不支持 | 不支持 |
| `isaacsim` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.11 worker; host Python paths are not inherited | 不支持 | 不支持 | 不支持 | 不支持 |

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix | Drake | mjwarp | Newton | SuperDex | Genesis | IsaacGym | IsaacSim |
|------------|------------|---|---|---|---|---|---|---|---|---|
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Tested | - | - | Configured | - | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Configured | - | Configured | Configured | Configured |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested | Tested | - | - | - | - | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `fr3_joint_target` (fr3 joint target) | - | - | - | - | - | Configured | - | - | - |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | Tested | Tested | - | - | - | - | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Registered | - | - | Registered | - | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | - | Registered | Registered | - | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | - | - | - | - | - | - |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested | Tested | - | - | - | - | - | - |
| SAC (torch) | `go2_joystick_flat` (Go2 joystick) | Registered | Registered | Tested | - | - | Registered | - | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Tested | - | Tested | Tested | Configured |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Configured | - | Configured | Configured | Configured |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Registered | - | Configured | - | - | - | - | - |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | Registered | - | - | - | - | - | - | - |
| SAC (torch) | `stewart_balance` (stewart balance) | Registered | Registered | Tested | - | - | - | - | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Registered | Registered | - | - | Registered | - | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Configured | Registered | - | Registered | Registered | Registered |
| FlashSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Configured | - | Configured | Registered | Registered |
| WarpSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | - | Tested | Registered | - | Registered | Registered | Registered |
| WarpSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | - | Tested | Registered | - | Registered | Registered | Registered |

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

IsaacGym 与 IsaacSim 分别使用专用 Python 3.8 和 Python 3.11 worker。worker
继承 CUDA 可见性命名空间，但不继承宿主 `PYTHONPATH` 或 `PYTHONHOME`。若
learner 当前 Torch CUDA 序号与 Isaac payload 整数序号不一致，构造前会失败；
同物理 GPU 的 CUDA IPC 还会在 worker 握手阶段再次校验。

在 macOS 与 ROCm 上请使用 CPU-authoritative host-bridge 后端。host-bridge
后端使用 CUDA Torch buffer 不代表 CUDA physics，也不代表 device-resident
lifecycle。
