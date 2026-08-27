# 硬件上的 G1 全身运动跟踪

::::{admonition} 硬件目标
:class: note
Unitree G1 人形机器人（29 自由度变体）。关节顺序来自任务 owner 的场景
（`src/unilab/assets/robots/g1/scene_flat.xml`，按 actuator 顺序）；在硬件上机前请
先核对该顺序与你的 SDK 电机索引是否一致。
::::

本指南说明 G1 运动跟踪策略在硬件上所期望的**观测与动作契约**。仓库不提供 G1 部署侧
运行时——硬件侧回路由你实现，本页告诉你它必须复现哪些内容。

## 0. 验证你的仿真侧检查点

```bash
# Replay the policy headlessly and produce a video.
uv run eval --algo sac --task g1_wbt_obs --sim mujoco --load-run -1 \
  --render-mode record
```

在视频中要关注：

- 被跟踪的各 body 跟随参考运动，没有大的不连续。
- 关节速度与动作保持有限且在预期范围内。
- 接触时序看起来与参考运动一致。

如果其中任何一项不对，请在硬件上机前修复仿真侧检查点。

## 1. 先选定 owner，再从 YAML 读取契约

硬件回路需要的每个字段都在任务 owner YAML 中声明。面向部署的 G1 owner 有：

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - Owner
  - Actor 观测宽度
  - 说明
* - `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml`
  - 514（H=5）
  - 带 proprio 历史、无状态估计：丢弃 `base_lin_vel` 与
    `motion_anchor_pos_b`，使用 pelvis IMU。
* - `conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml`
  - 154（H=1）
  - 单步 mimic actor 布局，逐关节 `action_scale` 列表。
```

::::{admonition} 观测宽度应从 env 读取，而不是照抄本表
:class: warning
Actor 观测宽度是 owner `noise_config` 各开关的函数——`g1_wbt_obs` 见
`src/unilab/envs/motion_tracking/g1/tracking_obs.py` 的 `_actor_obs_dim`，
deploy owner 见 `src/unilab/envs/motion_tracking/common/observations.py` 的
`mimic_actor_obs_dim`。如果 ONNX 输入宽度与硬件回路装配出的宽度不一致，那是契约
bug，而不是硬件调参问题。
::::

通过训练回放路径导出 `policy.onnx`：

```bash
uv run eval --algo sac --task g1_wbt_obs --sim mujoco --load-run -1
```

## 2. 观测契约

对 `g1_wbt_obs`，actor 观测按如下顺序装配（见
`src/unilab/envs/motion_tracking/g1/tracking_obs.py` 的 `_build_actor_obs`）。
单步参考项在前，随后是各 proprio 项的完整历史，按**最旧优先**展平：

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - 分组
  - 维度
  - 硬件上的来源
* - `command_joint_pos`
  - 29
  - 运动参考帧的关节位置
* - `command_joint_vel`
  - 29
  - 运动参考帧的关节速度
* - `motion_anchor_ori_b`
  - 6
  - 来自参考帧与机器人躯干帧的锚点朝向项
* - `gyro`
  - 每个历史步 3
  - IMU 陀螺仪项（`env.sensor.gyro`，该 owner 为 `pelvis_gyro`）
* - `joint_pos_rel`
  - 每个历史步 29
  - 测量到的关节位置减去 `stand` keyframe 的关节角
* - `dof_vel`
  - 每个历史步 29
  - 关节速度项
* - `last_actions`
  - 每个历史步 29
  - 上一步的原始 actor 输出
```

历史深度 `H` 即 `env.noise_config.obs_history_length`（该 owner 为 5）。逐项的
最旧优先顺序由 `tests/scripts/test_obs_alignment_g1_wbt.py` 守护；硬件侧必须镜像
该顺序，否则策略读到的是被置换过的向量。

## 3. 执行器接口

将 actor 输出映射为 `action * action_scale + default_angles`，然后在目标到达电机
驱动器之前钳制到场景的关节范围内。

- `action_scale` 即 owner YAML 中的 `env.control_config.action_scale`。它可能是
  **标量**（`g1_wbt_obs` 为 2.0），也可能是**逐关节列表**
  （`g1_motion_tracking_deploy` 为 29 项）。必须原样复现 owner 的形态——不要对列表
  取平均、取首项，也不要把标量广播到列表 owner 上。
- `default_angles` 是 owner 场景中 `stand` keyframe 的关节段。
- 关节限位与增益同样来自该场景 XML（`jnt_range`、position actuator 的
  `gainprm` / `biasprm`）。

训练侧直接施加目标，不做平滑。如果硬件抖动迫使你加入平滑，请先验证其 sim2sim
影响——每一步滞后都会把观测推离训练分布。

## 4. 参考运动同步

相位变量让策略能够跟踪一个外部提供的运动片段。在硬件上你需要一个墙钟 → 相位的映射，
它必须：

- **单调** —— 不向后跳跃。
- **可重启** —— 在通信抖动后仍能存活，不会在 `(sin φ, cos φ)` 中产生阶跃式
  不连续。
- **速率有界** —— 将 dφ/dt 钳制到策略训练时所用的值（运动加载器会记录这个值；加载
  `reference_motion.npz`）。

参见 `unilab.envs.motion_tracking.g1.motion_loader`，这是你应当在硬件上镜像的仿真侧
加载器。

## 5. 安全层

硬件侧：标准结构见 {doc}`7-safety_layers`。G1 的具体事项：

- 在应用 `action_scale` 之前拒绝非有限动作与形状不匹配。
- 用 owner 场景 XML 中的关节范围钳制生成的目标。
- 把看门狗、姿态监控以及操作员停止阈值保留在部署控制器中，并独立于策略对它们进行
  测试。

## 6. 闭环上机序列

1. **支架上站立**。机器人由龙门架吊挂。策略运行，但执行器关闭力矩。确认观测管线。
2. **使能力矩、手扶**。操作员护着机器人。策略指挥执行器。确认动作映射。
3. **龙门架支撑步态**。以半时间速率跟踪运动（dφ/dt 减半）。
4. **自由站立**。全速率，然后移除龙门架。

不要跳过仅观测阶段：轴顺序、关节顺序以及 `last_actions` 接线错误在这一阶段最容易被
抓出来。

## 7. 应记录什么

为每一步记录**完整的观测向量**、**完整的动作向量**与**墙钟**。把第一段硬件观测窗口
与用同一份 owner YAML 构建的仿真回合作对比——这个 diff 比任何奖励检查都更快定位
单位、坐标系与顺序错误。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`8-latency_budget`
- {doc}`7-safety_layers`
- {doc}`../../2-user_guide/4-tasks/2-motion_tracking`
