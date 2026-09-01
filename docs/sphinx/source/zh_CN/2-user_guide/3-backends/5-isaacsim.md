# IsaacSim 后端

UniLab 的 `isaacsim` 后端在独立的 Python 3.11 worker 进程中运行 IsaacSim
5.1.0 和 IsaacLab v2.3.0。主进程保留标准 `SimBackend` NumPy contract；管道
传输生命周期命令，共享内存传输批量状态。当前支持边界是已注册 G1 flat
task owner 的 headless physics 与 eval 专用原生渲染。支持矩阵仍将 PPO/SAC
owner 标为 `Configured`，不是 `Tested`：渲染协议有确定性 worker 测试，但在
当前可用 IsaacSim 主机上还没有完成真实 playback。

## 运行时边界

IsaacSim 5.1.0 只能在独立 Python 3.11 环境中运行，而 UniLab 主环境支持
Python 3.10--3.13。安装入口是 `scripts/tools/setup_isaacsim_env.sh`，默认安装
到 `$HOME/.unilab/isaacsim`，也可通过 `UNILAB_ISAACSIM_HOME` 修改。worker
启动通过 `OMNI_KIT_ACCEPT_EULA=1` 非交互接受 Kit EULA。

后端不会在主进程导入 Kit，并支持以下环境变量：

- `UNILAB_ISAACSIM_HOME`：运行时根目录。
- `UNILAB_ISAACSIM_PYTHON`：覆盖 worker 解释器路径。
- `OMNI_KIT_ACCEPT_EULA=1`：保持 worker 启动非交互。

预期目录为 `$UNILAB_ISAACSIM_HOME/venv/bin/python`、该 venv 下的
site-packages 和 library 目录，以及
`$UNILAB_ISAACSIM_HOME/IsaacLab` 下的 IsaacLab v2.3.0 源码。

渲染意图属于 worker 的冷路径 `INIT` 握手。训练不注入渲染模式，使用低开销的
headless、camera-disabled Kit experience。eval 在 Kit 启动前选择以下模式：

- `auto`：存在 `DISPLAY` 或 `WAYLAND_DISPLAY` 时使用 Kit 交互窗口，否则使用
  headless 离线录制。
- `interactive`：启动非 headless Kit；没有 display 环境变量时在 worker 启动前
  显式失败。
- `record`：启动带 IsaacLab RGB camera 的 headless rendering experience；
  `training.play_steps` 必须是有限值。
- `none`：不启动 viewer 或 camera，只执行策略 eval。

离线帧 contract 为 `(height, width, 3)`、`uint8`、连续且非均匀。错误帧或占位帧
会 fail closed，不会写出伪视频。IsaacSim owner 默认 1280 x 720，可在 env 创建前
通过 `env.isaacsim_render_width` 和 `env.isaacsim_render_height` 覆盖。

当前 worker 支持 MJCF 材质化、批量 articulation 状态、位置 target 步进和
masked root/joint reset，以及 Kit 原生 viewer 和 headless IsaacLab RGB camera。
contact-force sensor、reset/interval domain randomization 和 host pre-step
callback 仍不支持，并会 fail closed。

使用顶层 CLI 选择 backend 和 owner：

```bash
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
uv run eval --algo sac --task g1_walk_flat --sim isaacsim \
  --load-run <run-id> --render-mode record \
  training.play_steps=120 training.play_env_num=1 training.export_onnx=false
uv run eval --algo sac --task g1_walk_flat --sim isaacsim \
  --load-run <run-id> --render-mode interactive training.play_env_num=1
```

`record` 会在所选 run 目录写入 `play_video.mp4`。这些命令需要外部运行时和
NVIDIA CUDA 设备。仓库不声称已完成完整训练或稳定的原生 playback；该声明
需要 maintainer validation 记录。

## 当前运行时验证

已使用现有 SAC checkpoint 在 IsaacSim 5.1.0、IsaacLab v2.3.0、Kit 107.3.3、
Ubuntu 24.04.4、RTX 4090 和 NVIDIA driver 595.84 上分别执行有界 record eval
和 interactive eval。两条路径均在 `AppLauncher` 初始化期间、camera 或 viewer
创建之前崩溃；栈包含
`librtx.scenedb.plugin.so`、`libcarb.scenerenderer-rtx.plugin.so` 和
`libomni.hydra.rtx.plugin.so`，崩溃前有 EGL 初始化警告。最小 camera-enabled
`AppLauncher` probe 即使设置 `multi_gpu=False` 也失败。

这是真实 runtime blocker，不是 playback 成功证据。backend 保留渲染协议和
fail-closed 测试，支持矩阵仍为 `Configured`；真实 renderer 未初始化时不会生成
占位视频。

## 检查 Contract

```bash
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
OMNI_KIT_ACCEPT_EULA=1 \
uv run --active --no-project \
  scripts/tools/probe_isaacsim_contract.py \
  --model-file src/unilab/assets/robots/g1/scene_flat.xml \
  --num-envs 2 --steps 2 --device cuda:0 \
  --output /tmp/isaacsim-contract.json
```

该命令是有界的开发者探测，只在 cold-path 材质化阶段访问 XML/importer，适合
检查新安装的运行时；它不是训练或 playback 验证。

## Contract 矩阵

| UniLab contract | IsaacSim/IsaacLab 操作 | 实测结果 | 实现约束 |
|---|---|---|---|
| MJCF 场景材质化 | `isaaclab.sim.converters.MjcfConverter` | G1 MJCF 成功转换为 USD | headless worker 必须显式启用 `isaacsim.asset.importer.mjcf` |
| 批量 articulation | `Articulation` + `ArticulationCfg` | 2 个环境、29 joints、30 bodies | 材质化时按名称解析；不能假设 importer 顺序 |
| 四元数布局 | `robot.data.root_quat_w` / `body_quat_w` | `wxyz` | shared-memory 边界保持 `wxyz` |
| base 角速度 | `robot.data.root_ang_vel_w` | world frame | 公共 getter 保持 world frame；reset qvel 转换固定在 worker |
| partial reset | `write_root_pose_to_sim`、`write_root_velocity_to_sim`、`write_joint_state_to_sim`、`reset(env_ids)` | 只改变选中行，其他行 delta 为零 | 使用 masked batch write；拒绝重复或越界 id |
| 位置控制 | `set_joint_position_target`、`write_data_to_sim`、`SimulationContext.step` | 有界步数内首个 joint 移动 | `step(ctrl)` 传位置 target；gain/limit 在 cold path 显式材质化 |
| 状态 getter 边界 | `Articulation.data.*` tensor | getter 均为 batch，leading dimension 符合预期 | worker 将 tensor 拷贝到 host-owned shm；热路径不解析 asset |
| 渲染启动 | `AppLauncher` 冷路径模式选择 | mock worker 验证 none/record/interactive mode、尺寸和 graphics 握手 | env 材质化后不能切换 mode |
| 离线 RGB | IsaacLab `Camera` + `CameraCfg` | 协议测试验证视频写入，并拒绝错误 shape、dtype 或均匀帧；当前真实主机在 camera 创建前崩溃 | 要求有限步数；真实 playback 成功前支持等级保持 `Configured` |
| 交互窗口 | 非 headless Kit + `SimulationContext.set_camera_view` | 协议测试驱动一帧并将关闭窗口映射为 `RenderClosedError`；当前主机没有成功的有界 GUI 证据 | 显式 interactive 需要 display；无 display 时 `auto` 回退到 record |
| Domain randomization | IsaacLab manager/event API | 未探测 | 非空不支持 plan 必须 fail-closed |

Importer 返回的 joint/body 顺序与 MJCF 文档顺序不同（例如左右分支交错）。
worker 建立 name-to-index 映射并重排所有状态和控制数组；按位置假设顺序会
破坏 `SimBackend` index contract。完整 owner 和 capability 状态见
{doc}`../../5-reference/5-support_matrix`。
