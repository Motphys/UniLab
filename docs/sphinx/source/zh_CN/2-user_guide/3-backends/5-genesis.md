# Genesis 后端

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)（PyPI 分发名
`genesis-world`，仓库钉在 1.3.3）是一个 GPU 物理仿真器，UniLab 以
**进程内**方式使用它：`GenesisBackend` 在其上提供标准的 `SimBackend`
NumPy contract，物理与 learner 同进程运行——没有 worker 子进程，也没有
IPC。

当前状态：`GenesisBackend` 已实现并注册到 registry；`g1_walk_flat` 提供
PPO 与 SAC owner 配置（`conf/{ppo,sac}/task/g1_walk_flat/genesis.yaml`），
跨后端契约审计（`scripts/audit_sim2sim_contracts.py`）在两棵 algo 树上
覆盖 mujoco↔genesis（结论 TRANSFERABLE）。支持等级为 **experimental**：
现有证据是 registry + owner YAML + compose/contract 覆盖，以及真机
slow-lane env smoke
（`tests/envs/locomotion/g1/test_g1_owner_contract.py`：compose →
env 构造 → keyframe reset → 12 步有限稳定 → cleanup，覆盖 ppo 与 sac
两棵树），另有 SAC 短训练回路 smoke（64 envs / 3 iterations，经
learning_starts/updates_per_step 路径并保存 checkpoint）。尚未完成任何
训练验证，因此支持矩阵中这些 cell 标记为 `Configured`。

env 构造生命周期（#1383 已修复）：`ManagerBasedRlEnv` 构造期的 entity
校验在 env 的 `materialize()` 钩子之前读取状态 getter，因此 adapter 的
`materialize()` 是幂等且惰性触发的（首次状态访问时完成 `scene.build`，
与 IsaacGym 后端同模式）。adapter 的其余设计遵循
`scripts/tools/genesis_feasibility/REPORT.md` 的实测映射。

## 模型契约

Genesis 1.3.3 在 import 时丢弃三类 MJCF 特性（REPORT §3）：全局
`<option>` 块、`<keyframe>`、以及整个 `<sensor>` 块。adapter 在
materialize 冷路径补偿——热路径从不解析 XML：

- **全局 option**：MJCF `<option>` 承载的值由 owner YAML 显式重声明。
  对 g1（`<option integrator="implicitfast"
  timestep="0.006666666666666667"/>`），timestep 经现有
  `sim_dt` / `ctrl_dt` 链路传递，integrator 经
  `env.genesis_integrator: implicitfast` 声明。constraint solver、摩擦锥
  与 solver iterations 保持 Genesis 默认值（对应 `env.genesis_*` 字段为
  `None`）：MJCF option 块未声明它们，且 MuJoCo 隐式默认（PGS solver、
  pyramidal 摩擦锥）没有 Genesis 等价物。
- **keyframe**：materialize 时用 `mujoco` 包扫描一次并缓存，
  `default_keyframe_name: stand` 的 reset 行为不变。
- **传感器**：由 link 状态计算 MJCF 同名等价物，每个 accelerometer
  site 对应一个 `IMUSensor`（干净无噪声数据）。`data="found"` 的
  contact sensor 变为 per-link net contact force 阈值（1 N）——这是对
  geom 对 `found` 语义的近似，不是复现。
- **执行器**：`<position kp kv>` 执行器无损导入，用
  `control_dofs_position` 驱动；kp/kd reset 随机化受支持（DR 能力集只
  声明 per-env 往返实测的项：body mass、base mass、kp、kd，以及
  interval body force）。

## 前置条件

- Linux x86_64，装有 NVIDIA GPU 与驱动。可行性探针只验证了 `gs.gpu`
  通道；CPU backend 不是已验证的支持通道。
- 仓库钉定的 torch 窗口（x86_64 上为 torch 2.8）：探针中
  IMUSensor + contact-force 传感器组合在 torch 2.7 下崩溃（REPORT
  §3.4/§8）。
- 安装可选 extra（精确钉定 `genesis-world==1.3.3`）：

```bash
uv sync --extra genesis
```

## 训练与评估

训练通过标准 CLI 选择 genesis owner（owner YAML compose、registry
路由与 env 构造链路已在真机 slow-lane smoke 覆盖；尚无训练收敛证据）：

```bash
# PPO
uv run train --algo ppo --task g1_walk_flat --sim genesis

# SAC
uv run train --algo sac --task g1_walk_flat --sim genesis

# 小规模 smoke：64 个环境、只跑 3 个 iteration
uv run train --algo ppo --task g1_walk_flat --sim genesis \
    algo.num_envs=64 algo.max_iterations=3
```

## Playback 与渲染

Genesis 原生渲染是已声明能力，且在 `scene.build` 之后惰性挂载（训练热
路径不引入任何 viewer 依赖）：交互 viewer 是 post-build 挂上的
`genesis.vis.viewer.Viewer`，录制走 post-build 离屏 camera。owner 设置
`training.play_render_mode: auto`，按宿主解析：

- `auto`：有可及显示（`DISPLAY`/`WAYLAND_DISPLAY`）时打开**交互
  viewer**，否则降级为离屏**录制**。
- `interactive`：总是打开 viewer；无显示宿主上 `init_renderer` 报可
  操作错误。用户关闭 viewer 窗口即干净结束 playback（backend 抛
  `RenderClosedError`，play 循环视为正常退出）。
- `record`：无头离屏录制，要求有限的 `training.play_steps` 与输出
  路径；视频写到 checkpoint 运行目录的 `play_video.mp4`，帧率为
  `1/ctrl_dt`。
- `none`：安全跳过 playback（no-op）。

相机沿用仓库统一的球坐标 kwargs（`cam_distance` / `cam_elevation` /
`cam_azimuth`，`cam_tracking` 时跟随 `cam_tracking_env_idx` 的 root）。
评估已训练的 run：

```bash
uv run eval --algo sac --task g1_walk_flat --sim genesis \
    --load-run <run_dir_name> --render-mode record
```

`get_physics_state` 快照仍不声明：playback 直接驱动 live scene，无需
状态快照回放。

## 生命周期：一进程一次 `gs.init`

Genesis 的 init/destroy 循环每轮泄漏 200–450 MB host RSS（REPORT
§3.5 [9a]），因此 adapter 允许**每个进程恰好一次 `gs.init`**：session
销毁后再构造 backend 会 fail-closed 并给出明确错误。一个训练 run 一个
进程天然满足该约束；不要在长驻宿主进程里反复构建 genesis env。

## 未支持边界

以下能力 fail-closed（显式报错），而不是静默降级：

- **geom 名称契约**（`get_geom_names` 等）：不暴露。
- **生成式 terrain 与 height scanner**：`scene.terrain` 被拒绝；请选择
  flat owner YAML。
- **绝对值 geom 摩擦 DR**：上游 Genesis 只有 per-env 摩擦*比例* API，
  因此 `geom_friction` reset 随机化与 `get_geom_friction` fail-closed。
- **interval push / body-velocity-delta DR**：在构造/plan 时拒绝
  （`push_body_name`、push perturbation）。
- **同进程重复 `gs.init`**（见上文）。

## 跨后端迁移（sim2sim）

genesis owner 与 mujoco owner 在契约守卫
（`src/unilab/utils/sim2sim.py`）审计下保持 DENYLIST 一致（结论
TRANSFERABLE），同一 task 的 checkpoint 可跨后端使用。两侧都有原生
渲染，但实现不同：MuJoCo 从物理状态快照离线渲染，Genesis 用
post-build 挂载的 viewer/camera 驱动 live scene。
