# Newton 后端


[Newton](https://github.com/newton-physics/newton)（PyPI 分发名 `newton`，
仓库钉在 1.5.1）是基于 Warp 的 GPU 物理仿真器，UniLab 以**进程内**方式
使用它：`unisim.backend.newton.NewtonBackend` 在其上提供 public
`SimBackend` contract 的 tensor lane，物理与 learner 同进程运行——没有
worker 子进程，也没有 IPC。

当前状态：issue #2048 将 Newton 恢复进 tensor-only Manager runtime，但范围
仅限两个 canonical workload：

```bash
uv run train --algo sac --task g1_walk_flat --sim newton
uv run train --algo flashsac --task g1_motion_tracking --sim newton
```

owner 配置为 `src/unilab/conf/sac/task/g1_walk_flat/newton.yaml` 与
`src/unilab/conf/flashsac/task/g1_motion_tracking/newton.yaml`。其他
Newton owner 仍不在支持范围内，也不构成生产支持声明。public tensor lane
提供 device-resident state views、frame/contact sensors、state widths、
tensor stepping 与 authoritative selected-reset publication。Newton 不支持
的 reset randomization 在 owner 中显式禁用，绝不回退到 host composer。

Newton/Warp 遵循标准 CUDA 设备语义，单卡运行无需 `CUDA_VISIBLE_DEVICES`
钉卡。在多 GPU topology 中，rank 本地设备以 `newton_device="cuda:N"`
覆盖传入 spawn collector；uni_rl 的 collector 进程绑定为注入式，UniLab
注入的 `bind_backend_process_device_for_backend` 同时覆盖 MJWarp 与
Newton。

## 安装

Newton 运行时是 optional extra，钉定 Newton 1.5.1 与
MuJoCo-Warp 3.11 / Warp 1.16 系列：

```bash
# 源码 checkout：
uv sync --extra newton

# 从 PyPI 安装：
pip install "unilab[newton]"
```

该 extra 精确钉定 `newton==1.5.1`、`mujoco-warp==3.11.0`、
`mujoco==3.11.0`、`warp-lang==1.16.0`，并包含原生 ViewerGL 依赖
（`pyglet>=2.1.6,<3`、`imgui-bundle>=1.92.0`）。它与 `mujoco` extra
（`mujoco~=3.11.0`）和 `mjwarp` extra（`mujoco-warp~=3.11.0`、
`warp-lang==1.16.0`）共享同一条 MuJoCo 3.11 / MuJoCo-Warp 3.11 /
Warp 1.16 版本线，三个 extra 可以**组合进同一个环境**：

```bash
uv sync --extra mujoco --extra mjwarp --extra newton
```

前置条件：

- Linux，装有 NVIDIA GPU 与 CUDA 驱动。适配器的进程设备绑定
  （`unisim.backend.newton.runtime.bind_newton_process_device`）要求解析
  到可用的 CUDA Warp 设备，否则 fail-closed 报错；CPU 设备不是已验证的
  支持通道。
- Python `>=3.10`（仓库支持范围为 3.10–3.13）。

安装后可运行 unisim 仓库的 `scripts/check_newton_runtime.py` 做
metadata 级探针（加 `--import` 显式导入原生运行时）。在 UniLab 侧，缺少
Newton 运行时不是静默失败：顶层 CLI 在训练前检查 `newton` /
`newton` 模块，缺失时 fail-closed 并给出安装提示（`src/unilab/cli.py`
的 `_check_runtime_requirements`）。

## 训练与评估

上方 canonical tensor-runtime 命令选择 Newton owner。其他历史 owner 路径
在单独启用前仍不在支持范围内。

Newton/MuJoCo-Warp 3.11 要求显式的设备与存储容量，对应 owner YAML 中的
`env.*` 字段：

- `newton_device`：显式 backend CUDA placement。单设备 owner 使用
  `cuda:0`；多 GPU 训练会按 rank topology 覆盖该冷路径字段。
- `newton_nconmax` / `newton_njmax`：显式容量上界（canonical G1 owner
  为 320 / 512）。适配器在冷路径标定 solver 计数，容量不足时抛出明确的
  capacity 错误，绝不静默截断约束。
- `newton_capacity_check_steps`：容量检查步数（默认 1）。
- `newton_use_cuda_graph`：CUDA graph replay（默认 `true`）。支持 graph 的
  UniSim runtime 会在冷路径容量校准后捕获两个 Newton state 奇偶；CUDA
  驱动不满足条件或捕获失败时发出警告并回退 eager 执行。
  Runtime 证据必须报告 diagnostic 的实际 `enabled` 状态，不能根据请求推断。

## Playback 与渲染

canonical owner 选择 `training.play_render_mode: record`。Newton 通过
`newton` extra 内置的上游 `newton.viewer.ViewerGL` 渲染；若该安装不完整，
record 会回退 MuJoCo snapshot renderer。headless 离屏渲染仍需 OpenGL
context：无显示服务器的 Linux 主机设 `PYOPENGL_PLATFORM=egl`，Wayland 设
`PYOPENGL_PLATFORM=glx`。

```bash
uv run eval --algo sac --task g1_walk_flat --sim newton \
  --load-run <run_dir_name> --render-mode record
```

## 未支持边界

以下能力 fail-closed（显式报错或拒绝配置），而不是静默降级：

- **PD gain 运行时随机化**：Newton 当前拒绝该能力，owner 显式设置
  `events.pd_gains: null` 保持 fail-closed，直到适配器补齐。
- **原生渲染路径的相机参数**：native ViewerGL 路径忽略
  `camera_kwargs`（MuJoCo 快照路径仍生效）。

## 跨后端迁移（sim2sim）

newton owner 与 mujoco owner 在契约守卫
（`src/unilab/utils/sim2sim.py`）审计下保持 DENYLIST 一致（结论
TRANSFERABLE），同一 task 的 checkpoint 可跨后端使用。
