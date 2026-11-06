# Reward Porting

Map Legged Gym's `_reward_*` methods to `reward` terms in the owner YAML.
Manager-Based reward terms receive the env, read batched state through the
entity facade and managers, and return float32 `torch.Tensor`s of shape
`(num_envs,)` on the env device. Since 1.3.4 the Manager runtime is
Torch-only: the reward manager rejects non-Tensor returns with a type error.

## Tracking, smoothness, and joint limits

The `twist` command below must be defined in the owner's `env.commands`.
The tracking and action-rate entries follow
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`; the joint-limit and
termination entries illustrate existing helpers whose weights need task-specific
evaluation.

```yaml
reward:
  tracking_lin_vel:
    func: unilab.tasks.locomotion.common.sensor_reward_terms.track_lin_vel
    weight: 1.0
    params:
      sensor_name: local_linvel
      tracking_sigma: 0.5
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

`track_lin_vel` is a class term
(`src/unilab/tasks/locomotion/common/sensor_reward_terms.py`) that binds one
named 3-D velocity sensor at construction and computes
`exp(-||cmd_xy - v_xy||^2 / tracking_sigma)` against the `twist` command.
`tracking_sigma` is the direct denominator (default 0.25; the Go2 PPO owner
sets 0.5), and the velocity frame is whichever frame the bound sensor reports —
the Go2 owners bind `local_linvel`, a velocimeter on the IMU site. Both the
sensor and the command fail closed at construction when missing. When porting,
keep the sensor frame and the command dimensions consistent with the source
implementation.

`action_rate_l2` reads current and previous actions from the action manager.
`joint_pos_limits` reads soft joint limits through the entity facade.
Both return nonnegative costs; a negative `weight` supplies the penalty sign.
Do not negate the cost a second time.

## Actuator force readback

Declare one named `<actuatorfrc>` sensor per actuator in the robot or task
source XML. Each sensor reads MuJoCo's scalar actuation-space force, not the
joint-space generalized force:

```xml
<sensor>
  <actuatorfrc name="hip_force" actuator="hip"/>
  <actuatorfrc name="knee_force" actuator="knee"/>
</sensor>
```

A class-based Manager term declares the same names through
`tensor_sensor_names`. On the packed Torch path, read them in actuator order
and concatenate the `(num_envs, 1)` views when a matrix is needed:

```python
names = ("hip_force", "knee_force")
entity = env.scene["robot"]
views = env.scene.read_plan.sensor_tensor_views(entity, names).values
actuator_force = torch.cat([views[name] for name in names], dim=1)
```

The resulting shape is `(num_envs, num_actuators)`. Use the same packed views
for an L2 torque penalty instead of reconstructing force from PD gains.

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
- {doc}`../../6-how_to/2-custom_reward` — walkthrough for
  defining and verifying a new reward term end to end
- `src/unilab/envs/mdp/rewards.py`
- `src/unilab/tasks/locomotion/common/manager_terms.py`
- `src/unilab/tasks/locomotion/common/sensor_reward_terms.py`
- `src/unilab/tasks/locomotion/common/gait_terms.py`
