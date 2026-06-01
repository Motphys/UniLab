# 运动控制

运动控制任务注册在 `src/unilab/envs/locomotion/` 和
`src/unilab/envs/motion_tracking/` 中。`conf/` 下可用的 owner YAML
定义了哪些算法与后端组合是可运行的。

## 系列

- Go1：`go1_joystick_flat`、`go1_joystick_rough`
- Go2：`go2_joystick_flat`、`go2_joystick_rough`、`go2_handstand`、`go2_footstand`
- Go2W：`go2w_joystick_flat`、`go2w_joystick_rough`
- G1 行走：`g1_walk_flat`、`g1_walk_rough`
- G1 动作追踪：`g1_motion_tracking`、`g1_flip_tracking`、
  `g1_wall_flip_tracking`、`g1_climb_tracking`、`g1_box_tracking`
- Go2 机械臂：`go2_arm_manip_loco`

## 示例

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_rough --sim motrix training.no_play=true
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

查看支持矩阵以了解按 entrypoint、task owner 和 backend 划分的证据分级：
{doc}`../../5-reference/5-support_matrix`。

## Go2 FootStand

`go2_footstand` 是 Go2 前足站立任务，**仅支持 MuJoCo**。

- PPO 配置：`conf/ppo/task/go2_footstand/mujoco.yaml`
- 环境注册名：`Go2FootStand`（注册于 `sim_backend="mujoco"`）
- 环境实现：`src/unilab/envs/locomotion/go2/footstand.py`（继承 Go2 handstand 任务）
- Go2 模型 XML：`src/unilab/assets/robots/go2/go2.xml`

```bash
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run eval --algo ppo --task go2_footstand --sim mujoco --load-run -1
```

### 教师-学生训练流程

FootStand 的完整流程是三阶段教师-学生 pipeline；当前仓库里的 `go2_footstand` 配置
对应第一步，也就是教师策略的 PPO 训练入口：

1. **教师策略训练（特权观测）。** 教师策略可以使用特权观测（例如基座线速度），这些信息
   在仿真中可直接获得，但实机部署时不应直接依赖。训练时先让策略在较宽松的功率预算下学会
   前足站立（约 400 W），再通过课程学习逐步收紧到约 200 W，避免一开始就用低功率限制导致
   探索失败，同时让最终策略更接近可部署的能耗范围。
2. **学生策略蒸馏。** 把训练好的教师策略蒸馏到可部署的学生策略上，学生输入只保留实机可获得
   的观测，不依赖特权信息。蒸馏目标是在没有特权观测的条件下尽量复现教师策略的行为。
3. **学生策略微调。** 蒸馏后的学生策略继续做强化学习微调，使用统一损失目标：一部分来自与
   教师类似的奖励函数，另一部分来自教师正则项，约束学生策略不要过快偏离教师策略。这样既能
   保留学到的稳定动作，又能让学生策略适应自己的观测输入和部署约束。

### 观测口径

`Go2FootStand` 的策略（actor）网络观测使用 15 帧历史，每帧 45 维
（`_FOOTSTAND_FRAME_OBS_DIM = 45`）：

```text
linvel(3) + gyro(3) + gravity(3) + joint_position_delta(12) + joint_velocity(12) + last_action(12)
```

因此策略网络观测维度是 `45 * 15 = 675`。价值（critic）网络在这段历史观测后追加当前时刻的
特权观测尾部（`_FOOTSTAND_PRIVILEGED_TAIL_DIM = 49`）：

```text
gyro(3) + accelerometer(3) + linvel(3) + global_angvel(3) + dof_pos(12) + dof_vel(12) + torques(12) + height(1)
```

价值网络观测维度是 `675 + 49 = 724`。

### 奖励与终止项

默认奖励来自 `conf/ppo/task/go2_footstand/mujoco.yaml`。奖励权重包括站立 `height`、
`orientation`、`rear_feet_contact`、前腿目标角度（`tar`）、`action_rate`、
`dof_pos_limits`、`front_leg_motion`、`rear_leg_symmetry`、`knee_clearance`、
`upright_stability`、`stay_still`、`pose`，以及 `energy` 和 `dof_acc` 惩罚；
`termination` 与 `penalty_contact` 驱动终止/惩罚路径（前腿/前身体接触、低高度、坏朝向，
以及由 `energy_termination_threshold` 控制的高能耗截断）。

### 调参提示

- `env.obs_history_len`：策略观测历史长度，配置默认为 `15`。
- `env.energy_termination_threshold`：高能耗终止阈值，配置默认为 `200.0`。
- `env.domain_rand`：地面摩擦、连杆质量、机身质心、关节惯量和重置关节位置随机化。
- `reward.scales.height` / `orientation` / `rear_feet_contact`：站立姿态和后脚接触权重。

### 近风险检查

```bash
uv run pytest tests/envs/locomotion/test_go2_footstand.py tests/config/test_locomotion_params.py -q
```

如果改过 Go2 XML，至少确认 MuJoCo 能加载模型：

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2/go2.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```
