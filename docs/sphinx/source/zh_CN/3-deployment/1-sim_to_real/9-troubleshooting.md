# 仿真到真机故障排查

语言: 简体中文

一本 症状 → 可能原因 → 修复 的手册。当部署出岔子时，从这里开始。

## 高频关节振荡 / "嗡鸣"

| 可能原因 | 检查 | 修复 |
|---|---|---|
| PD 增益相对训练值过高 | 将驱动器 Kp/Kd 与 owner YAML 比较 | 匹配训练值；或用真实的 Kp/Kd DR 重新训练 |
| 训练中动作延迟过低 | 在 DR 中扫动 `torque_delay_ms` | 用 实测延迟 × 1.5 重新训练 |
| 速度噪声过低 | 比较仿真与硬件中的编码器 σ | 在 DR 中增大 `joint_vel_noise_std` |

## 站立 / 行走时漂移

| 可能原因 | 检查 | 修复 |
|---|---|---|
| 状态估计器速度偏置 | 记录 `base_lin_vel` 与真值（动作捕捉） | 调 KF 或切换到 HIM-PPO |
| IMU 偏置未标定 | 机器人静止，检查 `gyro_bias` | 在策略启动前运行 30 秒标定 |
| 足端接触分类错误 | 检查接触事件时间戳 | 对接触力阈值加滞回 |

## 策略在仿真中成功，在硬件上立即跌倒

几乎总是以下之一：

1. **关节顺序被调换。** 检查 `policy.onnx` 的输入宽度与你电机驱动器中的关节顺序。
   用 `unilab-export-scene` 导出训练时的关节顺序。
2. **动作缩放单位不匹配。** 策略输出未缩放的值；驱动器期望的是弧度，而你喂给它的
   是归一化的 [-1, 1]。在把目标发送给驱动器之前，应用 `deploy_config.yaml` 中的
   `action_scale` / 默认角度约定。
3. **观测布局不匹配。** 将 `deploy_config.yaml` 的 `obs_layout` 与训练 owner 比较，
   并在硬件上运行前用 `scripts/deploy/sim_prototype.py` 进行验证。

## Allegro / Sharpa 手内操作中方块掉落

| 可能原因 | 检查 | 修复 |
|---|---|---|
| 摩擦不匹配 | 真实方块与仿真 μ | 把摩擦 DR 扫得更宽，重新训练 |
| 抓取分布不匹配 | 记录操作员握持位姿 | 扩充抓取生成器 |
| 位姿估计器延迟 | 测量视觉管线毫秒数 | 添加观测滞后 DR |

## 策略在 MuJoCo 中成功但在 Motrix 中失败（或反之）

那是一个**仿真到仿真**问题，而不是仿真到真机。见
{doc}`../2-sim_to_sim/3-contact_and_friction_alignment` 与
{doc}`../2-sim_to_sim/4-reward_parity`。

## 提交 bug 报告前应捕获什么

当实验室微信群在晚上 11 点炸开时，保存以下内容，好让明天的排查只花 30 分钟而不是
4 小时：

- 完整的硬件轨迹（整段运行的 `obs / action / wall_clock`）。
- 用于训练的仿真侧 YAML：`runs/<run>/config.yaml`。
- `policy.onnx`，以及对于 G1 WBT 路径的 `deploy_config.yaml`。
- 一段使用**同一**种子的仿真回合视频：`eval --seed <same>
  --render-mode record`。
- 如果有的话，该运行所在 commit 与 `main` 之间的 `git diff`。

## 另请参阅

- {doc}`1-overview`
- {doc}`7-safety_layers`
- {doc}`8-latency_budget`

## Navigation

- Index: [文档](0-index.md)
