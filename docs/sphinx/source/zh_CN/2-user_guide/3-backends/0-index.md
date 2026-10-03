# 仿真后端

当前 tensor-only Manager runtime 暴露 `mujoco`、`mjwarp`、`genesis` 与
`newton`。用户命令通过 `--sim` 选择后端，并组合对应的 task owner YAML；不要
单独覆盖 `training.sim_backend`。

`motrix`、`drake`、`isaacgym`、`isaacsim` 与 `superdex` 适配器在 issue #1811
期间由 `unisim-core` 暂时搁置。历史页面仅用于 adapter 背景，不构成生产支持
声明。在提供新的 capability、parity 与支持矩阵证据之前，train/eval CLI 会
直接拒绝这些后端。

## Runtime 前置条件

- 任何使用 `--sim mujoco`、MuJoCo 回放或 MuJoCo 专属调试工具的运行，都需要
  `mujoco` extra 及其钉住的 `mjbatch` runtime。
- MJWarp 需要 `mjwarp` extra、NVIDIA CUDA；多 GPU 主机必须显式配置
  process-device topology。
- Genesis 需要 `genesis` extra 以及已验证的 Linux x86_64 GPU 路径。
- Newton 需要 `newton` extra 与 NVIDIA CUDA 设备；当前 tensor-native 支持范围
  为 SAC `g1_walk_flat` 和 FlashSAC `g1_motion_tracking`。

## OS 与 GPU 支持

| 后端 | 操作系统 | GPU |
| --- | --- | --- |
| MuJoCo | Linux / macOS / Windows | 不要求：CPU physics；离线回放可在 CPU 渲染 |
| MJWarp | Linux（已验证路径） | 要求：NVIDIA CUDA；单 GPU 主机默认使用当前 CUDA 设备 |
| Genesis | Linux x86_64 | 要求：NVIDIA GPU 与 driver；仅验证 `gs.gpu` channel |
| Newton | Linux | 要求：NVIDIA CUDA；selected-reset 通道为 device-resident |

后端设备要求与 learner 设备独立：MuJoCo 仍可以让 learner 使用 CUDA、ROCm、MPS
或 XPU。平台配置见 {doc}`../../1-getting_started/2-installation`。

## 选择后端

UniLab 通过 task owner config 选择仿真后端。常规用法通过 `--task` 与 `--sim`
选择；off-policy 命令中的算法保留在 `--algo`，不放在 `--task`。

### 快速选择

| 需求 | 优先选择 |
| --- | --- |
| 默认路径或 owner 覆盖最广 | MuJoCo |
| `scripts/play_viser.py` 等 MuJoCo 专属工具 | MuJoCo |
| Device-resident tensor owner | MJWarp、Genesis 或 Newton，且支持矩阵标记该组合为 supported |

支持矩阵由 registry、owner YAML 与测试生成；当前证据源见
{doc}`../../5-reference/5-support_matrix`。

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_motion_tracking --sim mjwarp
```

Owner YAML 位置：

- PPO / APPO：`src/unilab/conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy（SAC / FlashSAC）：`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`

选中的 owner YAML 会将 `training.sim_backend` 作为 identity field。

## 回放差异

- `--render-mode auto` 在 MuJoCo 路径导出 `play_video.mp4`。
- `--render-mode record` 录制但不打开交互窗口。
- `--render-mode viser` 在支持 physics-state playback 的后端（MuJoCo 与 MJWarp）
  中启动浏览器 viser viewer。
- `--render-mode none` 禁用回放。

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

## 支持证据

Task/backend/entrypoint 支持等级按证据分级。见
{doc}`../../5-reference/5-support_matrix`。

## 相关契约

- {doc}`Backend contract </en/4-developer_guide/2-contracts/2-backend_contract>`
- {doc}`Task owner contract </en/4-developer_guide/2-contracts/3-task_owner>`
- {doc}`Backend capability boundary ADR </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Registry bootstrap ADR </adr/ADR-0004-registry-bootstrap-contract>`

## unisim-core 边界

UniLab 的物理后端由独立的 `unisim-core` 发行版提供，Python namespace 为
`unisim`。例如：

```bash
uv sync --extra mujoco
uv run python -c "import unisim; print(unisim.ADAPTER_SPECS)"
```

`unisim` 不依赖 UniLab、Hydra 或训练组件。scoped
MuJoCo/MJWarp/Genesis/Newton 适配器与暂时搁置的适配器使用同一个 public
contract。缺失 proprietary SDK 或 GPU worker 时会产生明确的 cold-path
diagnostic；不会静默切换到另一个引擎。

```{toctree}
:hidden:

1-mujoco
5-genesis
7-newton
```
