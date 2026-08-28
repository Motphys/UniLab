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
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1 \
  --render-mode record
```

在视频中要关注：

- 被跟踪的各 body 跟随参考运动，没有大的不连续。
- 关节速度与动作保持有限且在预期范围内。
- 接触时序看起来与参考运动一致。

如果其中任何一项不对，请在硬件上机前修复仿真侧检查点。

## 1. 先导出，再从 owner YAML 读取契约

通过训练回放路径导出 `policy.onnx`，使用产出该检查点的同一任务 owner：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1
```

硬件回路需要的每个字段都在该 owner 的 YAML 中声明。宽度随 owner 而异——两个 G1
示例：

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - Owner
  - Actor 观测宽度
  - 说明
* - `conf/sac/task/g1_wbt_obs/mujoco.yaml`
  - 514
  - 无状态估计：`motion_anchor_pos_b` 与 `base_lin_vel` 置为 `null`，使用
    pelvis IMU，proprio 项带 `history_length: 5`。
* - `conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml`
  - 154
  - 单步 mimic actor 布局，按关节分组的 `scale` 正则映射。
```

::::{admonition} 观测宽度应从 composed config 读取，而不是照抄本表
:class: warning
Actor 观测宽度是 `env.observations.actor.terms` 下各项 `dim * history_length`
之和，再加上 motion command。置为 `null` 的项被丢弃。如果 ONNX 输入宽度与硬件回路
装配出的宽度不一致，那是契约 bug，而不是硬件调参问题。
::::

## 2. 观测契约

分项顺序与逐项历史来自你所用 owner 自己的 `env.observations.actor.terms`。以
`g1_wbt_obs` 为例，它声明了如下项；带 `history_length: 5` 的项在项内按**最旧优先**
展平，各项再按声明顺序拼接：

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - 项
  - 维度
  - 硬件上的来源
* - `motion_anchor_ori_b`
  - 6
  - 来自参考帧与机器人躯干帧的锚点朝向项
* - `base_ang_vel`
  - 每个历史步 3
  - IMU 陀螺仪（`params.sensor_name: pelvis_gyro`）
* - `joint_pos`
  - 每个历史步 29
  - 测量到的关节位置减去 `stand` keyframe 的关节角
* - `joint_vel`
  - 每个历史步 29
  - 关节速度项
* - `actions`
  - 每个历史步 29
  - 上一步的原始 actor 输出
```

motion command 在观测项之前贡献参考关节位置与速度（`29 + 29`）。逐项的最旧优先
顺序由 `tests/scripts/test_obs_alignment_g1_wbt.py` 守护；硬件侧必须镜像该顺序，
否则策略读到的是被置换过的向量。

## 3. 执行器接口

将 actor 输出映射为 `action * scale + default_angles`，然后在目标到达电机驱动器
之前钳制到场景的关节范围内。

- `scale` 即 `env.actions.joint_pos.scale`。它可能是**标量**（`g1_wbt_obs` 为
  `2.0`），也可能是按 actuator 解析的**正则 → 数值映射**
  （`g1_motion_tracking_deploy` 把关节名模式映射到不同数值）。必须原样复现 owner
  解析后的逐 actuator 向量——不要对映射取平均、取其中一项，也不要把标量广播到
  映射型 owner 上。
- `default_angles` 由 `use_default_offset: true` 决定，即 owner 场景中 `stand`
  keyframe 的关节段。
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

参见 `unilab.tasks.motion_tracking.common.motion_loader`，这是你应当在硬件上镜像的
仿真侧加载器。

## 5. 安全层

硬件侧：标准结构见 {doc}`7-safety_layers`。G1 的具体事项：

- 在应用动作缩放之前拒绝非有限动作与形状不匹配。
- 用 owner 场景 XML 中的关节范围钳制生成的目标。
- 把看门狗、姿态监控以及操作员停止阈值保留在部署控制器中，并独立于策略对它们进行
  测试。

## 6. 闭环上机序列

1. **支架上站立**。机器人由龙门架吊挂。策略运行，但执行器关闭力矩。确认观测管线。
2. **使能力矩、手扶**。操作员护着机器人。策略指挥执行器。确认动作映射。
3. **龙门架支撑步态**。以半时间速率跟踪运动（dφ/dt 减半）。
4. **自由站立**。全速率，然后移除龙门架。

不要跳过仅观测阶段：轴顺序、关节顺序以及 `actions` 接线错误在这一阶段最容易被
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
