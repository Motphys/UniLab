# 硬件上的 G1 全身运动跟踪

语言: 简体中文

::::{admonition} 硬件目标
:class: note
Unitree G1 人形机器人（29 自由度变体）。假定关节顺序与
`scripts/deploy/export_deploy_config.py` 从
`src/unilab/assets/robots/g1/scene_flat.xml` 导出的顺序一致；在硬件上机前请核对该
顺序。
::::

本指南讲解从一个收敛的 G1 运动跟踪策略到机器人上闭环运行之间的**最后一公里**。

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

如果其中任何一项不对，请在硬件上机前修复仿真侧检查点或部署契约。

## 1. 导出

使用训练回放路径导出 `policy.onnx`，然后用已提交的部署辅助工具导出 G1 WBT 部署配置
与运动二进制文件：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1

uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/export_motion_bin.py \
  --output logs/deploy/dance1.bin
```

部署侧原型消费如下文件：

```
runs/<run>/
└── policy.onnx
logs/deploy/
├── deploy_config.yaml
└── dance1.bin
```

## 2. 观测契约

对于已提交的 G1 WBT 部署辅助工具，观测布局会作为 `obs_layout` 导出到
`deploy_config.yaml`。`scripts/deploy/export_deploy_config.py` 是分段顺序的权威来源：

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
  - IMU 陀螺仪项
* - `joint_pos_rel`
  - 每个历史步 29
  - 测量到的关节位置减去 `default_angles`
* - `dof_vel`
  - 每个历史步 29
  - 关节速度项
* - `last_actions`
  - 29
  - 上一步的原始 actor 输出
```

导出脚本还会记录每个分段的 `history_length` 并校验总的 `obs_dim`。当 ONNX 输入宽度
与 `deploy_config.yaml` 的 `obs_dim` 不一致时，`scripts/deploy/sim_prototype.py` 会
拒绝运行。

## 3. 执行器接口

G1 部署原型将 actor 输出严格映射为：
`action * action_scale + default_angles`，然后钳制到 `joint_lower` /
`joint_upper`，并应用来自 `ema_alpha` 的 EMA 平滑。

- 动作 = 目标关节位置，按 `deploy_config.yaml` 中的 `action_scale` 项**缩放**。
- 在目标到达电机驱动器之前，将其钳制到生成的关节范围内。

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
- 用 `deploy_config.yaml` 中的 `joint_lower` / `joint_upper` 钳制生成的目标。
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

为每一步记录**完整的观测向量**、**完整的动作向量**与**墙钟**。在硬件上机前，通过
MuJoCo 部署原型用同一份 ONNX、部署配置与运动二进制进行验证：

```bash
uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

ONNX 输入宽度与 `deploy_config.yaml` 的 `obs_dim` 之间的不匹配是部署契约 bug，而不是
硬件调参问题。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`8-latency_budget`
- {doc}`7-safety_layers`
- {doc}`../../2-user_guide/4-tasks/2-motion_tracking`

## Navigation

- Index: [文档](0-index.md)
