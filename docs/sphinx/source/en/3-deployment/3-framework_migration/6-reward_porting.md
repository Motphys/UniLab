# Reward Porting

Map Legged Gym's `_reward_*` methods to `reward` terms in the owner YAML.
Manager-Based reward terms receive the env, read batched state through the
entity facade and managers, and return NumPy arrays of shape `(num_envs,)`.

## Tracking, smoothness, and joint limits

The `twist` command below must be defined in the owner's `env.commands`.
The tracking and action-rate entries follow
`src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml`; the joint-limit and
termination entries illustrate existing helpers whose weights need task-specific
evaluation.

```yaml
reward:
  tracking_lin_vel:
    func: unilab.tasks.locomotion.common.manager_terms.track_lin_vel_xy_exp
    weight: 1.0
    params:
      std: 0.5
      command_name: twist
  action_rate:
    func: unilab.envs.mdp.action_rate_l2
    weight: -0.005
  joint_limits:
    func: unilab.envs.mdp.joint_pos_limits
    weight: -1.0
  termination:
    func: unilab.envs.mdp.is_terminated
    weight: -1.0
```

`track_lin_vel_xy_exp` uses body-frame xy velocity error and computes
`exp(-error_squared / std**2)`. If the source denominator is `tracking_sigma`,
use `std = sqrt(tracking_sigma)` and preserve coordinate frames and command
dimensions.

`action_rate_l2` reads current and previous actions from the action manager.
`joint_pos_limits` reads soft joint limits through the entity facade.
Both return nonnegative costs; a negative `weight` supplies the penalty sign.
Do not negate the cost a second time.

## Contact-conditional rewards

Contact history is not a generic `state.prev_contact` field. Rewards needing
timers or edge detection use stateful manager terms and reset the corresponding
env rows on partial resets. Examples live in
`unilab.tasks.locomotion.common.gait_terms`: `feet_air_time` is a time-window
reward, while `foot_air_time` provides current airborne durations. These differ
from Legged Gym's first-contact bonus. Check trigger timing, time units, and
command gating, then compare per-term outputs on a fixed trajectory.

## Termination handling

`unilab.envs.mdp.is_terminated` reads the termination manager's non-timeout
termination mask. Give it a negative weight for a terminal penalty, and check
whether timeouts should contribute to the source task's penalty separately.

## See also

- {doc}`5-task_config_translation`
- `src/unilab/envs/mdp/rewards.py`
- `src/unilab/tasks/locomotion/common/manager_terms.py`
- `src/unilab/tasks/locomotion/common/gait_terms.py`
