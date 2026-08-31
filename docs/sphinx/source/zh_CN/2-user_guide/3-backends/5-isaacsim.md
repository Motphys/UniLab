# IsaacSim 后端可行性

本页记录 IsaacSim 5.1.0 / IsaacLab v2.3.0 运行时的实测证据。它是可行性
记录，不是支持声明；当前只通过 `scripts/tools/probe_isaacsim_contract.py`
执行有界探测。

## 运行时边界

IsaacSim 5.1.0 只能在独立 Python 3.11 环境中运行，而 UniLab 主环境支持
Python 3.10--3.13。安装入口是 `scripts/tools/setup_isaacsim_env.sh`，默认安装
到 `$HOME/.unilab/isaacsim`，也可通过 `UNILAB_ISAACSIM_HOME` 修改。worker
启动通过 `OMNI_KIT_ACCEPT_EULA=1` 非交互接受 Kit EULA。

首次 headless Kit 启动可能需要数分钟预热扩展和 shader cache。本页使用
IsaacLab headless `AppLauncher` 路径；在已验证的 595.x NVIDIA 驱动上，独立
GUI/full-app 路径可能在 `librtx.scenedb.plugin.so` 崩溃，因此暂不声明 GUI
回放能力（安装脚本中也记录了该限制）。

## 重现探测

```bash
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
OMNI_KIT_ACCEPT_EULA=1 \
uv run --active --no-project \
  scripts/tools/probe_isaacsim_contract.py \
  --model-file src/unilab/assets/robots/g1/scene_flat.xml \
  --num-envs 2 --steps 2 --device cuda:0 \
  --output /tmp/isaacsim-contract.json
```

该命令有界执行，只在 cold-path 材质化阶段访问 XML/importer；输出文件是下表
的证据来源。

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
worker 必须建立 name-to-index 映射并重排所有状态和控制数组；按位置假设顺序
会破坏 `SimBackend` index contract。

## 当前范围

本页和探测脚本不会注册 `isaacsim`、添加 task owner，也不声明训练/play 支持。
这些改动属于 issue #1369 下的实现切片，必须分别补充 conformance 和运行时证据。
