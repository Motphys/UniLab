# Reward Term

内置 reward term，由源码整理。所有 term 每个环境返回一个 float32 标量；
reward manager 会乘上 YAML 中的 `weight`。框架级 term 在
`unilab.envs.mdp.<name>` 处重导出。

:::{only} api_ref
```{raw} html
<ul>
  <li>权威签名见 <a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API 参考</a>、
      <a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API 参考</a>、
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API 参考</a>。</li>
</ul>
```
:::

:::{only} prose_only
权威签名见源码树：
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp)、
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks)。
:::

## 框架级（`unilab.envs.mdp.rewards`）

源码：`src/unilab/envs/mdp/rewards.py`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `is_alive`
  - `unilab.envs.mdp.rewards.is_alive`（函数）
  - —
  - 环境未发生非超时终止时恒为 1.0。
* - `is_terminated`
  - `unilab.envs.mdp.rewards.is_terminated`（函数）
  - —
  - 非超时终止时为 1.0；配合负权重惩罚失败。
* - `root_height`
  - `unilab.envs.mdp.rewards.root_height`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 世界系根高度（z），作为标量奖励或度量。
* - `joint_vel_l2`
  - `unilab.envs.mdp.rewards.joint_vel_l2`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 所选关节速度的平方和。
* - `action_rate_l2`
  - `unilab.envs.mdp.rewards.action_rate_l2`（函数）
  - —
  - 原始策略动作一阶差分的平方和。
* - `action_acc_l2`
  - `unilab.envs.mdp.rewards.action_acc_l2`（函数）
  - —
  - 原始策略动作二阶差分的平方和。
* - `flat_orientation_l2`
  - `unilab.envs.mdp.rewards.flat_orientation_l2`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 世界重力向量在根 body 系下 xy 分量的平方和；惩罚机身不水平。
* - `upright`
  - `unilab.envs.mdp.rewards.upright`（函数）
  - `std: float`（必填，> 0）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 高斯奖励 `exp(-||pg_xy||^2 / std^2)`，用于保持单个所选 body 竖直；
    要求恰好选中一个 body。仅平地形式。
* - `track_linear_velocity`
  - `unilab.envs.mdp.rewards.track_linear_velocity`（函数）
  - `std: float`（必填，> 0）；`command_name: str`（必填）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `exp(-(||cmd_xy - v_xy||^2 + v_z^2) / std^2)`，速度取机体系；假设
    z 向指令为零。
* - `track_angular_velocity`
  - `unilab.envs.mdp.rewards.track_angular_velocity`（函数）
  - `std: float`（必填，> 0）；`command_name: str`（必填）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `exp(-((cmd_yaw - w_z)^2 + ||w_xy||^2) / std^2)`；奖励 yaw 角速度
    跟踪，同时抑制 roll/pitch 角速度。
* - `body_angular_velocity_penalty`
  - `unilab.envs.mdp.rewards.body_angular_velocity_penalty`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 单个所选 body 的 roll/pitch 角速度平方和；要求恰好选中一个 body。
* - `joint_pos_limits`
  - `unilab.envs.mdp.rewards.joint_pos_limits`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 所选关节超出软限位的线性（未平方）违例量之和。
* - `posture`
  - `unilab.envs.mdp.rewards.posture`（类）
  - `std: dict[str, float]`（必填；关节名正则 → 逐关节 std）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 对所选关节计算 `exp(-mean((q - q_default)^2 / std^2))`。
* - `variable_posture`
  - `unilab.envs.mdp.rewards.variable_posture`（类）
  - `std_standing` / `std_walking` / `std_running: dict[str, float]`
    （必填）；`command_name: str`（必填）；
    `walking_threshold: float = 0.5`；`running_threshold: float = 1.5`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 带速度自适应容差的 `posture`：按总指令速度（平面模长加 yaw 绝对值）
    在站立/行走/奔跑三档 std 表之间选择。
```

## Locomotion —— 通用 term

### `unilab.tasks.locomotion.common.manager_terms`

源码：`src/unilab/tasks/locomotion/common/manager_terms.py`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `base_height_l2`
  - `unilab.tasks.locomotion.common.manager_terms.base_height_l2`（类）
  - `target_height: float`（必填，m）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 世界系根高度与 `target_height` 的平方误差。
* - `alive`
  - `unilab.tasks.locomotion.common.manager_terms.alive`（函数）
  - —
  - 每步恒定 1.0，无条件。
* - `joint_deviation_l1`
  - `unilab.tasks.locomotion.common.manager_terms.joint_deviation_l1`（类）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 所选关节相对默认位姿的绝对偏差之和（L1）。
* - `stand_still_l1`
  - `unilab.tasks.locomotion.common.manager_terms.stand_still_l1`（类）
  - `command_name: str`（必填）；`command_threshold: float = 0.1`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 仅在指令模长低于 `command_threshold` 时计入的
    `joint_deviation_l1`。
* - `feet_phase_contact`
  - `unilab.tasks.locomotion.common.manager_terms.feet_phase_contact`
    （类）
  - `sensor_names: list[str]`（必填；4 个互不相同的接触传感器）；
    `frequency: float = 2.0`（Hz）；
    `phase_offsets: list[float] = (0.0, 0.5, 0.5, 0.0)`；
    `contact_threshold: float = 0.1`；`stance_threshold: float = 0.6`；
    `command_name: str | None = None`；`command_threshold: float = 0.0`
  - 实测接触与步态相位期望站立段一致的脚所占比例。
* - `feet_phase_swing_height`
  - `unilab.tasks.locomotion.common.manager_terms.feet_phase_swing_height`
    （类）
  - `sensor_names: list[str]`（必填；4 个 3 维位置传感器）；
    `frequency: float = 2.0`；`phase_offsets = (0.0, 0.5, 0.5, 0.0)`；
    `target_height: float = 0.1`（m）；`kernel: float = 0.01`（> 0）；
    `swing_start: float = 0.6`；`command_name: str | None = None`；
    `command_threshold: float = 0.0`
  - 摆动相内对每只脚给 `exp(-(z - target_height)^2 / kernel)`，再对脚
    取平均。
* - `feet_air_while_standing`
  - `unilab.tasks.locomotion.common.manager_terms.feet_air_while_standing`
    （类）
  - `sensor_names: list[str]`（必填；4 个接触传感器）；
    `command_name: str`（必填）；`command_threshold: float = 0.1`；
    `contact_threshold: float = 0.1`
  - 指令模长不超过 `command_threshold`（站立）时统计无接触的脚数。
```

### `unilab.tasks.locomotion.common.gait_terms`

源码：`src/unilab/tasks/locomotion/common/gait_terms.py`。带
`sensor_groups` 的 term 为每只脚绑定一组具名接触传感器；组内任一传感器
超过 `contact_threshold` 即视为该脚接触。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `feet_air_time`
  - `unilab.tasks.locomotion.common.gait_terms.feet_air_time`（类）
  - `sensor_groups: list[list[str]]`（必填；每脚一组）；
    `contact_threshold: float = 0.0`；`threshold_min: float = 0.05`（s）；
    `threshold_max: float = 0.5`（s，须大于 `threshold_min`）；
    `command_name: str | None = None`；`command_threshold: float = 0.5`
  - 统计当前腾空时间落在奖励窗口内的脚数；设置 `command_name` 时按指令
    门控。
* - `feet_swing_height`
  - `unilab.tasks.locomotion.common.gait_terms.feet_swing_height`（类）
  - `sensor_groups: list[list[str]]`（必填）；`target_height: float`
    （必填，> 0，m）；`command_name: str`（必填）；
    `command_threshold: float`（必填）；`contact_threshold: float = 0.0`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`（每个传感器组
    对应一个脚部 body）
  - 腾空期间跟踪峰值脚高；首次着地时对每只脚计
    `(peak / target_height - 1)^2`。
* - `feet_slip`
  - `unilab.tasks.locomotion.common.gait_terms.feet_slip`（类）
  - `sensor_groups: list[list[str]]`（必填）；`command_name: str`
    （必填）；`command_threshold: float = 0.01`；
    `contact_threshold: float = 0.0`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 接触状态下脚部 xy 速度平方和（打滑惩罚）。
* - `feet_clearance`
  - `unilab.tasks.locomotion.common.gait_terms.feet_clearance`（类）
  - `target_height: float`（必填，m）；
    `command_name: str | None = None`；
    `command_threshold: float = 0.01`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`（脚部 body）
  - 对每只脚累加 `|z - target_height|` 并按平面脚速加权。
* - `self_collision_cost`
  - `unilab.tasks.locomotion.common.gait_terms.self_collision_cost`（类）
  - `sensor_groups: list[list[str]]`（必填）；
    `contact_threshold: float = 0.0`
  - 报告接触的自碰撞传感器组数量。
* - `angular_momentum_penalty`
  - `unilab.tasks.locomotion.common.gait_terms.angular_momentum_penalty`
    （类）
  - `sensor_name: str`（必填；一个 3 维传感器）
  - 从该具名传感器读取的整机角动量平方范数。
```

### `unilab.tasks.locomotion.common.sensor_reward_terms`

源码：`src/unilab/tasks/locomotion/common/sensor_reward_terms.py`。每个
term 绑定机器人的一个具名 3 维 XML 传感器。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `track_lin_vel`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.track_lin_vel`
    （类）
  - `sensor_name: str`（必填；线速度传感器）；
    `tracking_sigma: float = 0.25`（> 0）；`command_name: str = "twist"`
  - `exp(-||cmd_xy - v_xy||^2 / tracking_sigma)`。
* - `track_ang_vel`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.track_ang_vel`
    （类）
  - `sensor_name: str`（必填；陀螺仪传感器）；
    `tracking_sigma: float = 0.25`（> 0）；`command_name: str = "twist"`
  - `exp(-(cmd_yaw - w_z)^2 / tracking_sigma)`。
* - `lin_vel_z`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.lin_vel_z`（类）
  - `sensor_name: str`（必填）
  - 竖直方向（z）线速度平方。
* - `ang_vel_xy`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.ang_vel_xy`（类）
  - `sensor_name: str`（必填）
  - roll/pitch 角速度平方和。
* - `orientation`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.orientation`（类）
  - `sensor_name: str`（必填；朝上向量传感器）
  - 朝上向量水平分量平方和；惩罚偏离竖直。
```

## Locomotion —— G1（`unilab.tasks.locomotion.g1.manager_terms`）

源码：`src/unilab/tasks/locomotion/g1/manager_terms.py`。步态类 term 共享
参数 `frequency: float = 1.5`（Hz）、`init_mode: str = "offset_phase"`
（`"offset_phase"` 或 `"independent"`）、`swing_height: float = 0.09`
（m）、`tracking_sigma: float = 0.008`（> 0）、
`min_forward_speed: float = 0.0`、`command_name: str = "twist"`；并绑定具名
传感器 `left_foot_pos` / `right_foot_pos` 与 `pelvis_local_linvel`（接触类
term 额外绑定 `left_foot_contact_0..3` / `right_foot_contact_0..3`）。同一
env 上的步态 term 必须共享 `frequency` 与 `init_mode`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `feet_phase`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase`（类）
  - 共享步态参数（见上）
  - 相对三次 Bézier 逐脚高度目标计算
    `exp(-(err_left^2 + err_right^2) / tracking_sigma)`，并按前向速度
    门控。
* - `feet_phase_contrast`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase_contrast`（类）
  - 共享步态参数
  - `exp(-((h_left - h_right) - (t_left - t_right))^2 / tracking_sigma)`；
    奖励左右脚高差与相位目标的对比一致。
* - `feet_phase_contact`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase_contact`（类）
  - 共享步态参数（见上）
  - 实测接触与期望站立相位逐脚匹配率的均值，按前向速度门控。
* - `feet_double_stance`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_double_stance`（类）
  - 共享步态参数（见上）
  - 前向指令激活时惩罚双脚同时着地。
* - `feet_air_time`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_air_time`（类）
  - 共享步态参数（见上）
  - 统计当前腾空时间落在硬编码 (0.05, 0.5) s 窗口内的脚数。
* - `forward_progress`
  - `unilab.tasks.locomotion.g1.manager_terms.forward_progress`（类）
  - `command_name: str = "twist"`
  - `min(clamp(v_x, min=0) / clamp(cmd_x, min=1e-6), 1)`；相对指令速度的
    前向进度。
* - `under_speed`
  - `unilab.tasks.locomotion.g1.manager_terms.under_speed`（类）
  - `command_name: str = "twist"`
  - 相对欠速量 `clamp(cmd_x - clamp(v_x, min=0), min=0) /
    clamp(cmd_x, min=1e-6)`。
* - `penalty_feet_ori`
  - `unilab.tasks.locomotion.g1.manager_terms.penalty_feet_ori`（类）
  - —
  - 两只脚坐标系四元数 x、y 分量的平方和；惩罚脚底不水平。
* - `penalty_close_feet_xy`
  - `unilab.tasks.locomotion.g1.manager_terms.penalty_close_feet_xy`（类）
  - `threshold: float = 0.15`（m）
  - 双脚水平距离小于 `threshold` 时计 `(dist_xy - threshold)^2`。
* - `weighted_pose`
  - `unilab.tasks.locomotion.g1.manager_terms.weighted_pose`（类）
  - `pose_weights: list[float]`（必填；与所选关节一一对应）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 关节相对默认位姿的加权平方偏差和。
* - `upper_body_pose`
  - `unilab.tasks.locomotion.g1.manager_terms.upper_body_pose`（类）
  - `pose_weights: list[float]`（必填，至少 12 项）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 将前 12 个（腿部）权重置零后的 `weighted_pose`。
* - `G1WalkRewardPack`
  - `unilab.tasks.locomotion.g1.manager_terms.G1WalkRewardPack`（类；配置
    `G1WalkRewardPackCfg`）
  - `command_name: str = "twist"`；`entity_name: str = "robot"`；
    `tracking_lin_sigma: float = 0.25`；
    `tracking_ang_sigma: float = 0.25`；`gait_frequency: float = 1.5`；
    `gait_init_mode: str = "offset_phase"`；
    `swing_height: float = 0.09`；`feet_phase_sigma: float = 0.04`；
    `min_forward_speed: float = 0.0`；
    `pose_weights: tuple[float, ...] = 29 个 1.0`
  - 一次张量读取内融合计算 G1 行走奖励族：线速度/角速度跟踪、角速度 xy
    与姿态惩罚、动作变化率、加权位姿、脚部姿态惩罚、门控相位跟踪与
    alive。以 `_target_` + `func` 声明并经 `reward_pack_names` 接线
    （见 `src/unilab/conf/sac/task/g1_walk_flat/mjwarp.yaml`）。
```

## Motion tracking（`unilab.tasks.motion_tracking`）

源码：`src/unilab/tasks/motion_tracking/common/manager_terms.py` 与
`src/unilab/tasks/motion_tracking/g1/manager_terms.py`。所有 term 按
`command_name` 解析一个 `MotionCommand` term；`std` 参数必须为正。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `motion_global_anchor_position_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_global_anchor_position_error_exp`（函数）
  - `command_name: str`（必填）；`std: float`（必填，> 0）
  - 世界系下 `exp(-||p_anchor_motion - p_anchor_robot||^2 / std^2)`。
* - `motion_global_anchor_orientation_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_global_anchor_orientation_error_exp`（函数）
  - `command_name: str`（必填）；`std: float`（必填，> 0）
  - `exp(-theta^2 / std^2)`，theta 为动作与机器人 anchor 姿态间的最短
    路径四元数夹角。
* - `MotionRelativeBodyPositionErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionRelativeBodyPositionErrorExp`（类）
  - `command_name: str`（必填）；`std: float`（必填，> 0）；
    `body_names: tuple[str, ...] | None = None`（指令所跟踪 body 的
    子集）
  - 对 anchor 相对系下各 body 位置误差计算
    `exp(-sum_b ||dp_b||^2 / (N * std^2))`。
* - `MotionRelativeBodyOrientationErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionRelativeBodyOrientationErrorExp`（类）
  - `command_name: str`（必填）；`std: float`（必填，> 0）；
    `body_names: tuple[str, ...] | None = None`
  - 对 anchor 相对系下各 body 姿态误差计算
    `exp(-sum_b theta_b^2 / (N * std^2))`。
* - `MotionGlobalBodyLinearVelocityErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionGlobalBodyLinearVelocityErrorExp`（类）
  - `command_name: str`（必填）；`std: float`（必填，> 0）；
    `body_names: tuple[str, ...] | None = None`
  - 对世界系各 body 线速度误差计算
    `exp(-sum_b ||dv_b||^2 / (N * std^2))`。
* - `MotionGlobalBodyAngularVelocityErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionGlobalBodyAngularVelocityErrorExp`（类）
  - `command_name: str`（必填）；`std: float`（必填，> 0）；
    `body_names: tuple[str, ...] | None = None`
  - 对世界系各 body 角速度误差计算
    `exp(-sum_b ||dw_b||^2 / (N * std^2))`。
* - `motion_joint_position_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_position_error_exp`（函数）
  - `command_name: str`（必填）；`std: float`（必填，> 0）
  - `exp(-mean((q_motion - q_robot)^2) / std^2)`。
* - `motion_joint_velocity_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_velocity_error_exp`（函数）
  - `command_name: str`（必填）；`std: float`（必填，> 0）
  - `exp(-mean((dq_motion - dq_robot)^2) / std^2)`。
* - `joint_pos_limits`
  - `unilab.tasks.motion_tracking.common.manager_terms.joint_pos_limits`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 软限位违例量的平方和；任务局部实现，与框架级同名线性项不同。
* - `UndesiredBodyContacts`
  - `unilab.tasks.motion_tracking.common.manager_terms.UndesiredBodyContacts`（类）
  - `command_name: str`（必填）；`threshold: float`（必填，m）；
    `body_names: tuple[str, ...] | None = None`
  - 统计世界系 z 低于 `threshold` 的所选 body 数量。
* - `joint_acc_l2`
  - `unilab.tasks.motion_tracking.g1.manager_terms.joint_acc_l2`（类）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 有限差分关节加速度 `(dq_t - dq_t-1) / env.step_dt` 的平方和；带
    reset 感知。当前未被随仓库发布的 owner YAML 引用。
* - `joint_torque_l2`
  - `unilab.tasks.motion_tracking.g1.manager_terms.joint_torque_l2`（类）
  - `action_name: str = "joint_pos"`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 位置控制器力矩估计 `kp * (target - q) - kd * dq` 的平方和；要求
    存在目标顺序与执行器顺序一致的 `MotionJointPositionAction`。当前
    未被随仓库发布的 owner YAML 引用。
```
