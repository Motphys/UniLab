# IsaacSim 后端

UniLab 的 `isaacsim` 后端在独立的 Python 3.11 worker 进程中运行 IsaacSim
5.1.0 和 IsaacLab v2.3.0。主进程保留标准 `SimBackend` NumPy contract；管道
传输生命周期命令，共享内存传输批量状态。当前支持边界是已注册 G1 flat
task owner 的 headless physics。支持矩阵将 PPO/SAC owner 标为
`Configured`，不是 `Tested`。

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

首次 headless Kit 启动可能需要数分钟预热扩展和 shader cache。在已验证的
595.x NVIDIA 驱动上，独立 GUI/full-app 路径可能在
`librtx.scenedb.plugin.so` 崩溃；因此本后端不声明 GUI、camera capture 或
native playback，worker 始终使用 IsaacLab headless `AppLauncher` 路径。

当前 worker 支持 MJCF 材质化、批量 articulation 状态、位置 target 步进和
masked root/joint reset。contact-force sensor、reset/interval domain
randomization、host pre-step callback、GUI rendering、camera capture 和
native playback 均不支持，并会 fail closed。

使用顶层 CLI 选择 backend 和 owner：

```bash
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
uv run eval --algo sac --task g1_walk_flat --sim isaacsim --render-mode none
```

这些命令需要外部运行时和 NVIDIA CUDA 设备。仓库没有声称上述命令已完成
完整训练或 playback 验证；该声明需要 maintainer validation 记录。

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
| 渲染 | IsaacLab headless `AppLauncher` | 本探测不声明 | 独立驱动验证前保持 GUI/native rendering fail-closed |
| Domain randomization | IsaacLab manager/event API | 未探测 | 非空不支持 plan 必须 fail-closed |

Importer 返回的 joint/body 顺序与 MJCF 文档顺序不同（例如左右分支交错）。
worker 建立 name-to-index 映射并重排所有状态和控制数组；按位置假设顺序会
破坏 `SimBackend` index contract。完整 owner 和 capability 状态见
{doc}`../../5-reference/5-support_matrix`。
