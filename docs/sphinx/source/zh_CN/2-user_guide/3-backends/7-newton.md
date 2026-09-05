# Newton 后端

[Newton](https://github.com/newton-physics/newton)（PyPI 分发名 `newton`，
仓库钉在 1.5.1）是基于 Warp 的 GPU 物理仿真器，UniLab 以**进程内**方式
使用它：`unisim.backend.newton.NewtonBackend` 在其上提供标准的
`SimBackend` NumPy contract，物理与 learner 同进程运行——没有 worker
子进程，也没有 IPC。

当前状态：Newton 已在 owner/config 边界完成接线（UniLab PR
[#1511](https://github.com/unilabsim/UniLab/pull/1511)，适配器由 unisim
仓库的 `unisim.backend.newton` 提供）。`g1_walk_flat` 提供 PPO 与 SAC
owner 配置（`src/unilab/conf/{ppo,sac}/task/g1_walk_flat/newton.yaml`，
registry 注册 `G1WalkFlat` 的 `newton` 后端），跨后端契约审计
（`scripts/audit_sim2sim_contracts.py`）在两棵 algo 树上覆盖
mujoco↔newton（结论 TRANSFERABLE）。支持等级：SAC 为 **Tested**——
2026-09-06 在 RTX 4090 / torch 2.8.0+cu128 / newton 1.5.1 / mujoco-warp
3.11 上完成完整 5000-iteration 训练（reward/mean 6.68 → 242.3，
episode length → 983，约 43k steps/s，wall 4m25s），并对
`model_5000.pt` 做了 record 回放验证；PPO 保持 **Configured**（现有
证据限于 owner 配置、compose/contract 检查和 fail-closed 的
runtime/import 边界，尚无训练或回放验证结论）。

## 安装

Newton 运行时是**隔离的 optional extra**，钉定 Newton 1.5.1 与
MuJoCo-Warp 3.11 / Warp 1.16 系列：

```bash
# 源码 checkout：
uv sync --extra newton

# 从 PyPI 安装：
pip install "unilab[newton]"
```

该 extra 精确钉定 `newton==1.5.1`、`mujoco-warp==3.11.0`、
`mujoco==3.11.0`、`warp-lang==1.16.0`，与历史 `mjwarp` extra
（`mujoco-warp==3.10.0.3`）和 `mujoco` extra 使用的 MuJoCo 版本线不同，
因此 `pyproject.toml` 通过 uv `conflicts` 声明三者**互斥**：Newton 不能
与 `mujoco` 或 `mjwarp` extra 装进同一环境。需要对比 MuJoCo / MJWarp 与
Newton 时，请为 Newton 单独建一个环境。

前置条件：

- Linux，装有 NVIDIA GPU 与 CUDA 驱动。适配器的进程设备绑定
  （`unisim.backend.newton.runtime.bind_newton_process_device`）要求解析
  到可用的 CUDA Warp 设备，否则 fail-closed 报错；CPU 设备不是已验证的
  支持通道。
- Python `>=3.10`（仓库支持范围为 3.10–3.13）。

安装后可运行 unisim 仓库的 `scripts/check_newton_runtime.py` 做
metadata 级探针（加 `--import` 显式导入原生运行时）。在 UniLab 侧，缺少
Newton 运行时不是静默失败：顶层 CLI 在训练前检查 `newton` /
`mujoco_warp` / `mujoco` / `warp` 模块，缺失时 fail-closed 并给出安装
提示（`src/unilab/cli.py` 的 `_check_runtime_requirements`）。

## 训练与评估

训练通过标准 CLI 选择 newton owner：

```bash
# PPO
uv run train --algo ppo --task g1_walk_flat --sim newton

# SAC
uv run train --algo sac --task g1_walk_flat --sim newton
```

Newton/MuJoCo-Warp 3.11 要求显式的设备与存储容量，对应 owner YAML 中的
`env.*` 字段：

- `newton_device`：owner 默认 `null` 是有意设计——进程设备绑定器
  （`src/unilab/base/process_device.py`）在 materialization 前注入
  rank 本地的 CUDA 设备。显式设置时必须是非空 CUDA 设备字符串。
- `newton_nconmax` / `newton_njmax`：显式容量上界（g1 owner 为
  320 / 512）。适配器在冷路径标定 solver 计数，容量不足时抛出明确的
  capacity 错误，绝不静默截断约束。
- `newton_capacity_check_steps`：容量检查步数（默认 1）。

## Playback 与渲染

Newton 后端尚无原生 UniLab viewer：owner 设置
`training.play_render_mode: record`，回放走录制 / 无头路径；请求原生
交互式渲染器会 fail-closed。

```bash
uv run eval --algo ppo --task g1_walk_flat --sim newton \
    --load-run <run_dir_name> --render-mode record

uv run eval --algo sac --task g1_walk_flat --sim newton \
    --load-run <run_dir_name> --render-mode record
```

## 未支持边界

以下能力 fail-closed（显式报错或拒绝配置），而不是静默降级：

- **PD gain 运行时随机化**：Newton 当前拒绝该能力，owner 显式设置
  `events.pd_gains: null` 保持 fail-closed，直到适配器补齐。
- **原生交互式渲染**：见上文 Playback 一节。

## 跨后端迁移（sim2sim）

newton owner 与 mujoco owner 在契约守卫
（`src/unilab/utils/sim2sim.py`）审计下保持 DENYLIST 一致（结论
TRANSFERABLE），同一 task 的 checkpoint 可跨后端使用。
