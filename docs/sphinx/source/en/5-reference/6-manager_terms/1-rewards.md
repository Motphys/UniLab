# Reward Terms

Built-in reward terms, compiled from source. All terms return one float32
value per environment; the reward manager multiplies by the YAML `weight`.
Framework-level terms are re-exported at `unilab.envs.mdp.<name>`.

:::{only} api_ref
```{raw} html
<ul>
  <li>Authoritative signatures: <a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API reference</a>,
      <a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API reference</a>,
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API reference</a></li>
</ul>
```
:::

:::{only} prose_only
Authoritative signatures live in the source tree:
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp),
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks).
:::

## Framework (`unilab.envs.mdp.rewards`)

Source: `src/unilab/envs/mdp/rewards.py`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `is_alive`
  - `unilab.envs.mdp.rewards.is_alive` (function)
  - —
  - 1.0 while the environment has not reached a non-timeout termination.
* - `is_terminated`
  - `unilab.envs.mdp.rewards.is_terminated` (function)
  - —
  - 1.0 on non-timeout terminations; pair with a negative weight to penalize
    failure.
* - `root_height`
  - `unilab.envs.mdp.rewards.root_height` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - World-frame root height (z) as a scalar reward or metric.
* - `joint_vel_l2`
  - `unilab.envs.mdp.rewards.joint_vel_l2` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Sum of squared velocities over the selected joints.
* - `action_rate_l2`
  - `unilab.envs.mdp.rewards.action_rate_l2` (function)
  - —
  - Sum of squares of the first difference of raw policy actions.
* - `action_acc_l2`
  - `unilab.envs.mdp.rewards.action_acc_l2` (function)
  - —
  - Sum of squares of the second difference of raw policy actions.
* - `flat_orientation_l2`
  - `unilab.envs.mdp.rewards.flat_orientation_l2` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Squared xy components of the world gravity vector expressed in the root
    body frame; penalizes non-flat base orientation.
* - `upright`
  - `unilab.envs.mdp.rewards.upright` (function)
  - `std: float` (required, > 0);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Gaussian reward `exp(-||pg_xy||^2 / std^2)` for keeping one selected body
    upright; requires exactly one selected body. Flat-ground form only.
* - `track_linear_velocity`
  - `unilab.envs.mdp.rewards.track_linear_velocity` (function)
  - `std: float` (required, > 0); `command_name: str` (required);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `exp(-(||cmd_xy - v_xy||^2 + v_z^2) / std^2)` with body-frame velocity;
    assumes commanded z is zero.
* - `track_angular_velocity`
  - `unilab.envs.mdp.rewards.track_angular_velocity` (function)
  - `std: float` (required, > 0); `command_name: str` (required);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `exp(-((cmd_yaw - w_z)^2 + ||w_xy||^2) / std^2)`; rewards commanded yaw
    rate while keeping roll/pitch rates near zero.
* - `body_angular_velocity_penalty`
  - `unilab.envs.mdp.rewards.body_angular_velocity_penalty` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Squared roll/pitch angular velocity of one selected body; requires exactly
    one selected body.
* - `joint_pos_limits`
  - `unilab.envs.mdp.rewards.joint_pos_limits` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Sum of joint position violations beyond the soft limits (linear, not
    squared).
* - `posture`
  - `unilab.envs.mdp.rewards.posture` (class)
  - `std: dict[str, float]` (required; joint-name regex to per-joint std);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `exp(-mean((q - q_default)^2 / std^2))` over the selected joints.
* - `variable_posture`
  - `unilab.envs.mdp.rewards.variable_posture` (class)
  - `std_standing` / `std_walking` / `std_running: dict[str, float]`
    (required); `command_name: str` (required);
    `walking_threshold: float = 0.5`; `running_threshold: float = 1.5`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `posture` with speed-dependent tolerance: the std map is selected from the
    total command speed (planar norm plus yaw magnitude).
```

## Locomotion — shared terms

### `unilab.tasks.locomotion.common.manager_terms`

Source: `src/unilab/tasks/locomotion/common/manager_terms.py`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `base_height_l2`
  - `unilab.tasks.locomotion.common.manager_terms.base_height_l2` (class)
  - `target_height: float` (required, m);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Squared error between world-frame root height and `target_height`.
* - `alive`
  - `unilab.tasks.locomotion.common.manager_terms.alive` (function)
  - —
  - Constant 1.0 reward every step, unconditional.
* - `joint_deviation_l1`
  - `unilab.tasks.locomotion.common.manager_terms.joint_deviation_l1` (class)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Sum of absolute joint displacement from the default pose over the selected
    joints.
* - `stand_still_l1`
  - `unilab.tasks.locomotion.common.manager_terms.stand_still_l1` (class)
  - `command_name: str` (required); `command_threshold: float = 0.1`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `joint_deviation_l1` masked to environments whose command norm is below
    `command_threshold`.
* - `feet_phase_contact`
  - `unilab.tasks.locomotion.common.manager_terms.feet_phase_contact` (class)
  - `sensor_names: list[str]` (required; 4 unique contact sensors);
    `frequency: float = 2.0` (Hz);
    `phase_offsets: list[float] = (0.0, 0.5, 0.5, 0.0)`;
    `contact_threshold: float = 0.1`; `stance_threshold: float = 0.6`;
    `command_name: str | None = None`; `command_threshold: float = 0.0`
  - Fraction of feet whose measured contact matches the expected stance
    portion of the gait phase.
* - `feet_phase_swing_height`
  - `unilab.tasks.locomotion.common.manager_terms.feet_phase_swing_height`
    (class)
  - `sensor_names: list[str]` (required; 4 3-D position sensors);
    `frequency: float = 2.0`; `phase_offsets = (0.0, 0.5, 0.5, 0.0)`;
    `target_height: float = 0.1` (m); `kernel: float = 0.01` (> 0);
    `swing_start: float = 0.6`; `command_name: str | None = None`;
    `command_threshold: float = 0.0`
  - `exp(-(z - target_height)^2 / kernel)` for each foot during its swing
    phase, averaged over feet.
* - `feet_air_while_standing`
  - `unilab.tasks.locomotion.common.manager_terms.feet_air_while_standing`
    (class)
  - `sensor_names: list[str]` (required; 4 contact sensors);
    `command_name: str` (required); `command_threshold: float = 0.1`;
    `contact_threshold: float = 0.1`
  - Counts feet without contact while the command norm is at or below
    `command_threshold`.
```

### `unilab.tasks.locomotion.common.gait_terms`

Source: `src/unilab/tasks/locomotion/common/gait_terms.py`. Terms with
`sensor_groups` bind one named contact-sensor group per foot; any sensor in
the group above `contact_threshold` counts as foot contact.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `feet_air_time`
  - `unilab.tasks.locomotion.common.gait_terms.feet_air_time` (class)
  - `sensor_groups: list[list[str]]` (required; one group per foot);
    `contact_threshold: float = 0.0`; `threshold_min: float = 0.05` (s);
    `threshold_max: float = 0.5` (s, > `threshold_min`);
    `command_name: str | None = None`; `command_threshold: float = 0.5`
  - Counts feet whose current air time sits inside the rewarded window;
    command-gated when `command_name` is set.
* - `feet_swing_height`
  - `unilab.tasks.locomotion.common.gait_terms.feet_swing_height` (class)
  - `sensor_groups: list[list[str]]` (required); `target_height: float`
    (required, > 0, m); `command_name: str` (required);
    `command_threshold: float` (required); `contact_threshold: float = 0.0`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")` (one foot body per
    sensor group)
  - Tracks the peak foot height while airborne; at first contact charges
    `(peak / target_height - 1)^2` per landing foot.
* - `feet_slip`
  - `unilab.tasks.locomotion.common.gait_terms.feet_slip` (class)
  - `sensor_groups: list[list[str]]` (required); `command_name: str`
    (required); `command_threshold: float = 0.01`;
    `contact_threshold: float = 0.0`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Sum of squared xy foot velocity while the foot is in contact.
* - `feet_clearance`
  - `unilab.tasks.locomotion.common.gait_terms.feet_clearance` (class)
  - `target_height: float` (required, m); `command_name: str | None = None`;
    `command_threshold: float = 0.01`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")` (foot bodies)
  - Sum over feet of `|z - target_height|` weighted by planar foot speed.
* - `self_collision_cost`
  - `unilab.tasks.locomotion.common.gait_terms.self_collision_cost` (class)
  - `sensor_groups: list[list[str]]` (required);
    `contact_threshold: float = 0.0`
  - Count of monitored self-collision sensor groups reporting contact.
* - `angular_momentum_penalty`
  - `unilab.tasks.locomotion.common.gait_terms.angular_momentum_penalty`
    (class)
  - `sensor_name: str` (required; one 3-D sensor)
  - Squared norm of the whole-body angular momentum read from the named
    sensor.
```

### `unilab.tasks.locomotion.common.sensor_reward_terms`

Source: `src/unilab/tasks/locomotion/common/sensor_reward_terms.py`. Each
term binds one named 3-D XML sensor of the robot.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `track_lin_vel`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.track_lin_vel` (class)
  - `sensor_name: str` (required; linear-velocity sensor);
    `tracking_sigma: float = 0.25` (> 0); `command_name: str = "twist"`
  - `exp(-||cmd_xy - v_xy||^2 / tracking_sigma)`.
* - `track_ang_vel`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.track_ang_vel` (class)
  - `sensor_name: str` (required; gyro sensor);
    `tracking_sigma: float = 0.25` (> 0); `command_name: str = "twist"`
  - `exp(-(cmd_yaw - w_z)^2 / tracking_sigma)`.
* - `lin_vel_z`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.lin_vel_z` (class)
  - `sensor_name: str` (required)
  - Squared vertical (z) linear velocity.
* - `ang_vel_xy`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.ang_vel_xy` (class)
  - `sensor_name: str` (required)
  - Squared roll/pitch angular velocity sum.
* - `orientation`
  - `unilab.tasks.locomotion.common.sensor_reward_terms.orientation` (class)
  - `sensor_name: str` (required; up-vector sensor)
  - Squared horizontal up-vector components; penalizes deviation from
    upright.
```

## Locomotion — G1 (`unilab.tasks.locomotion.g1.manager_terms`)

Source: `src/unilab/tasks/locomotion/g1/manager_terms.py`. The gait terms
share the parameters `frequency: float = 1.5` (Hz),
`init_mode: str = "offset_phase"` (`"offset_phase"` or `"independent"`),
`swing_height: float = 0.09` (m), `tracking_sigma: float = 0.008` (> 0),
`min_forward_speed: float = 0.0`, and `command_name: str = "twist"`; they bind
the named sensors `left_foot_pos` / `right_foot_pos` and
`pelvis_local_linvel` (contact terms additionally bind
`left_foot_contact_0..3` / `right_foot_contact_0..3`). Gait terms on one env
must share `frequency` and `init_mode`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `feet_phase`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase` (class)
  - Shared gait parameters (see above)
  - `exp(-(err_left^2 + err_right^2) / tracking_sigma)` against cubic-Bézier
    per-foot height targets, gated by forward speed.
* - `feet_phase_contrast`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase_contrast` (class)
  - Shared gait parameters
  - `exp(-((h_left - h_right) - (t_left - t_right))^2 / tracking_sigma)`;
    rewards left/right foot-height contrast against the phase targets.
* - `feet_phase_contact`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_phase_contact` (class)
  - Shared gait parameters (see above)
  - Mean per-foot match between measured contact and the expected stance
    phase, gated by forward speed.
* - `feet_double_stance`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_double_stance` (class)
  - Shared gait parameters (see above)
  - Penalizes double-stance contact while a forward command is active.
* - `feet_air_time`
  - `unilab.tasks.locomotion.g1.manager_terms.feet_air_time` (class)
  - Shared gait parameters (see above)
  - Counts feet whose current air time sits in the hardcoded (0.05, 0.5) s
    window.
* - `forward_progress`
  - `unilab.tasks.locomotion.g1.manager_terms.forward_progress` (class)
  - `command_name: str = "twist"`
  - `min(clamp(v_x, min=0) / clamp(cmd_x, min=1e-6), 1)`; forward progress
    relative to commanded speed.
* - `under_speed`
  - `unilab.tasks.locomotion.g1.manager_terms.under_speed` (class)
  - `command_name: str = "twist"`
  - Relative shortfall `clamp(cmd_x - clamp(v_x, min=0), min=0) /
    clamp(cmd_x, min=1e-6)`.
* - `penalty_feet_ori`
  - `unilab.tasks.locomotion.g1.manager_terms.penalty_feet_ori` (class)
  - —
  - Sum of the squared x and y quaternion components of both foot frames;
    penalizes non-flat foot orientations.
* - `penalty_close_feet_xy`
  - `unilab.tasks.locomotion.g1.manager_terms.penalty_close_feet_xy` (class)
  - `threshold: float = 0.15` (m)
  - `(dist_xy - threshold)^2` while the feet are closer than `threshold` in
    the horizontal plane.
* - `weighted_pose`
  - `unilab.tasks.locomotion.g1.manager_terms.weighted_pose` (class)
  - `pose_weights: list[float]` (required; one per selected joint);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Weighted sum of squared joint deviation from the default pose.
* - `upper_body_pose`
  - `unilab.tasks.locomotion.g1.manager_terms.upper_body_pose` (class)
  - `pose_weights: list[float]` (required, at least 12 entries);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - `weighted_pose` with the twelve leg-joint weights forced to zero.
* - `G1WalkRewardPack`
  - `unilab.tasks.locomotion.g1.manager_terms.G1WalkRewardPack` (class; cfg
    `G1WalkRewardPackCfg`)
  - `command_name: str = "twist"`; `entity_name: str = "robot"`;
    `tracking_lin_sigma: float = 0.25`; `tracking_ang_sigma: float = 0.25`;
    `gait_frequency: float = 1.5`; `gait_init_mode: str = "offset_phase"`;
    `swing_height: float = 0.09`; `feet_phase_sigma: float = 0.04`;
    `min_forward_speed: float = 0.0`;
    `pose_weights: tuple[float, ...] = 29 x 1.0`
  - Fused canonical G1 walk reward family evaluated in one tensor carrier
    read: linear/angular tracking, angular-xy and orientation penalties,
    action rate, weighted pose, foot-orientation penalty, gated phase
    tracking, and alive. Declared with `_target_` +
    `func` and wired to `reward_pack_names` (see
    `src/unilab/conf/sac/task/g1_walk_flat/mjwarp.yaml`).
```

## Motion tracking (`unilab.tasks.motion_tracking`)

Sources: `src/unilab/tasks/motion_tracking/common/manager_terms.py` and
`src/unilab/tasks/motion_tracking/g1/manager_terms.py`. All terms resolve a
`MotionCommand` term by `command_name`; `std` parameters must be positive.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `motion_global_anchor_position_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_global_anchor_position_error_exp`
    (function)
  - `command_name: str` (required); `std: float` (required, > 0)
  - `exp(-||p_anchor_motion - p_anchor_robot||^2 / std^2)` in the world frame.
* - `motion_global_anchor_orientation_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_global_anchor_orientation_error_exp`
    (function)
  - `command_name: str` (required); `std: float` (required, > 0)
  - `exp(-theta^2 / std^2)` with the shortest-path quaternion angle between
    motion and robot anchor orientation.
* - `MotionRelativeBodyPositionErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionRelativeBodyPositionErrorExp` (class)
  - `command_name: str` (required); `std: float` (required, > 0);
    `body_names: tuple[str, ...] | None = None` (subset of the command's
    tracked bodies)
  - `exp(-sum_b ||dp_b||^2 / (N * std^2))` over anchor-relative body
    positions.
* - `MotionRelativeBodyOrientationErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionRelativeBodyOrientationErrorExp` (class)
  - `command_name: str` (required); `std: float` (required, > 0);
    `body_names: tuple[str, ...] | None = None`
  - `exp(-sum_b theta_b^2 / (N * std^2))` over anchor-relative body
    orientations.
* - `MotionGlobalBodyLinearVelocityErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionGlobalBodyLinearVelocityErrorExp` (class)
  - `command_name: str` (required); `std: float` (required, > 0);
    `body_names: tuple[str, ...] | None = None`
  - `exp(-sum_b ||dv_b||^2 / (N * std^2))` over world-frame body linear
    velocities.
* - `MotionGlobalBodyAngularVelocityErrorExp`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionGlobalBodyAngularVelocityErrorExp` (class)
  - `command_name: str` (required); `std: float` (required, > 0);
    `body_names: tuple[str, ...] | None = None`
  - `exp(-sum_b ||dw_b||^2 / (N * std^2))` over world-frame body angular
    velocities.
* - `motion_joint_position_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_position_error_exp` (function)
  - `command_name: str` (required); `std: float` (required, > 0)
  - `exp(-mean((q_motion - q_robot)^2) / std^2)`.
* - `motion_joint_velocity_error_exp`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_velocity_error_exp` (function)
  - `command_name: str` (required); `std: float` (required, > 0)
  - `exp(-mean((dq_motion - dq_robot)^2) / std^2)`.
* - `joint_pos_limits`
  - `unilab.tasks.motion_tracking.common.manager_terms.joint_pos_limits` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Sum of squared soft-limit violations; task-local squared variant, distinct
    from the linear framework term of the same name.
* - `UndesiredBodyContacts`
  - `unilab.tasks.motion_tracking.common.manager_terms.UndesiredBodyContacts` (class)
  - `command_name: str` (required); `threshold: float` (required, m);
    `body_names: tuple[str, ...] | None = None`
  - Count of selected bodies whose world-frame z drops below `threshold`.
* - `joint_acc_l2`
  - `unilab.tasks.motion_tracking.g1.manager_terms.joint_acc_l2` (class)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Squared finite-difference joint acceleration `(dq_t - dq_t-1) /
    env.step_dt`, summed; reset-aware. Not referenced by the shipped owner
    YAMLs.
* - `joint_torque_l2`
  - `unilab.tasks.motion_tracking.g1.manager_terms.joint_torque_l2` (class)
  - `action_name: str = "joint_pos"`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Squared position-controller torque estimate `kp * (target - q) -
    kd * dq`, summed; requires a `MotionJointPositionAction` whose target
    order matches the actuator order. Not referenced by the shipped owner
    YAMLs.
```
