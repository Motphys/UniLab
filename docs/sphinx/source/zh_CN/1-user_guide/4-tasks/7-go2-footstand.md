# Go2 FootStand

语言: 简体中文

本页记录 `go2_footstand` 的 PPO 任务入口、配置位置、整体训练流程和近风险验证命令。

## 任务

- PPO：`go2_footstand`
- 支持后端：MuJoCo

## 默认命令

```bash
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run eval --algo ppo --task go2_footstand --sim mujoco --load-run -1
```

## 配置入口

- PPO 配置：`conf/ppo/task/go2_footstand/mujoco.yaml`
- 环境注册名：`Go2FootStand`
- 环境实现：`src/unilab/envs/locomotion/go2/footstand.py`
- Go2 模型 XML：`src/unilab/assets/robots/go2/go2.xml`

## 教师-学生训练流程

FootStand 的完整训练流程分为三步：先训练带特权观测的教师策略，再把教师策略蒸馏到可部署的
学生策略，最后对学生策略继续做强化学习微调。当前仓库里的 `go2_footstand` 配置对应第一步，
也就是教师策略的 PPO 训练入口。

1. 教师策略训练

   教师策略可以使用特权观测，其中包括基座线速度等仿真中可以直接获得、但实机部署时不应直接
   依赖的信息。训练时先让策略在较宽松的功率预算下学会前足站立动作，再通过课程学习逐步收紧
   功率约束：从约 400 W 过渡到约 200 W。这样可以避免一开始就使用低功率限制导致探索失败，同时
   让最终策略更接近实机可承受的能耗范围。

2. 学生策略蒸馏

   教师策略训练完成后，将它蒸馏到一个可部署的学生策略上。学生策略的输入只保留实机可获得的
   观测，不能依赖特权信息。蒸馏阶段的目标是让学生策略在没有特权观测的条件下，尽量复现教师策略
   的行为。

3. 学生策略微调

   蒸馏后的学生策略还需要继续强化学习微调。微调时使用一个统一的损失目标：一部分来自与教师训练
   类似的奖励函数，另一部分来自教师策略正则项，用来约束学生策略不要过快偏离教师策略。这样既能
   保留教师策略学到的稳定动作，又能让学生策略继续适应自己的观测输入和部署约束。

## 仿真到实机注意点

- 特权观测只服务于教师策略训练；可部署的学生策略不应直接读取基座线速度、仿真全局角速度或其他
  只在仿真中可靠存在的传感量。
- 功率课程学习的目标是先保证动作能学出来，再逐步适应 200 W 左右的部署约束，避免低功率限制过早
  压缩探索空间。
- 学生策略微调阶段的教师正则项用于稳定行为分布，但权重过大时会限制学生策略对实机观测噪声、
  延迟和接触差异的适应。
- 实机侧要重点检查惯性测量单元噪声、关节速度噪声、接触状态、执行器带宽和功率估计口径；这些
  偏差会直接影响前足站立的能耗、姿态和终止条件。

## 观测口径

`Go2FootStand` 的策略网络观测使用 15 帧历史，每帧 45 维：

```text
linvel(3) + gyro(3) + gravity(3) + joint_position_delta(12) + joint_velocity(12) + last_action(12)
```

因此策略网络观测维度是 `45 * 15 = 675`。价值网络会在这段历史观测后追加当前时刻的特权观测：

```text
gyro(3) + accelerometer(3) + linvel(3) + global_angvel(3) + dof_pos(12) + dof_vel(12) + torques(12) + height(1)
```

价值网络观测维度是 `675 + 49 = 724`。

## 奖励与终止项

该任务的默认奖励来自 `conf/ppo/task/go2_footstand/mujoco.yaml`，主要包括：

- 站立高度、朝向、后脚接触、前腿目标角度；
- 动作变化率、关节限位、前腿运动、后腿对称、膝盖离地高度、静止约束；
- 能耗和关节加速度惩罚；
- 前腿/前身体接触、低高度、坏朝向、能量阈值等终止或惩罚路径。

## 调参提示

- `env.obs_history_len`：策略观测历史长度，当前配置默认为 15。
- `env.energy_termination_threshold`：高能耗终止阈值。
- `env.domain_rand`：摩擦、连杆质量、机身质心、关节惯量和重置关节位置随机化。
- `reward.scales.height`、`orientation`、`rear_feet_contact`：站立姿态和后脚接触权重。

## 近风险检查

```bash
uv run pytest tests/envs/locomotion/test_go2_footstand.py tests/config/test_locomotion_params.py -q
```

如果改过 Go2 XML，至少确认 MuJoCo 能加载模型：

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2/go2.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

## 关联入口

- 训练规则：看 [03 训练指南](../3-training.md)
- 任务总索引：看 [D 任务索引](1-task-index.md)

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [Go2 Arm Manip Loco](6-go2-arm-manip-loco.md)
