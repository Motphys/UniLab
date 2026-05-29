# Reward Porting

Reward terms are where most porting bugs hide. This cookbook captures the
common terms and their UniLab idiom.

## Pattern: linear / quadratic tracking error

```python
# Legged Gym
def _reward_tracking_lin_vel(self):
    err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-err / self.cfg.rewards.tracking_sigma)

# UniLab
def reward_tracking_lin_vel(self, state):
    err = np.sum((state.commands[:, :2] - state.base_lin_vel[:, :2]) ** 2, axis=1)
    return np.exp(-err / self.cfg.tracking_sigma)
```

Notes:

- UniLab reward terms operate on a `state` *batch* (NumPy on CPU); no
  per-env loop, no `torch`.
- Return per-env scalar reward (shape `(n_envs,)`).

## Pattern: contact-conditional bonus

```python
def reward_feet_air_time(self, state):
    contact = state.foot_contact     # bool, (n_envs, n_feet)
    air_time = state.last_air_time   # float, (n_envs, n_feet)
    first_contact = contact & ~state.prev_contact
    reward = (air_time - self.cfg.air_time_threshold) * first_contact
    return reward.sum(axis=1)
```

Notes:

- UniLab's `state` carries `prev_contact` so you don't need to manage
  edge detection yourself. See
  `unilab.envs.locomotion.common.rewards`.

## Pattern: action smoothness penalty

```python
def reward_action_rate(self, state):
    return -np.sum((state.action - state.prev_action) ** 2, axis=1)
```

Already a stock helper in `unilab.envs.locomotion.common.rewards`.

## Pattern: posture penalty

```python
def reward_dof_pos_limits(self, state):
    lower = self.cfg.dof_pos_lower
    upper = self.cfg.dof_pos_upper
    deviation = (
        np.maximum(0, lower - state.dof_pos) +
        np.maximum(0, state.dof_pos - upper)
    )
    return -np.sum(deviation, axis=1)
```

## Termination handling

UniLab separates **terminal signal** from **terminal penalty**. The env's
`terminations()` returns a boolean mask; the reward registry can include
a `termination_penalty` term that consumes it.

```python
def reward_termination(self, state):
    return -state.termination.astype(np.float32) * self.cfg.termination_penalty
```

## See also

- {doc}`task_config_translation`
- `unilab.training.reward`
- `unilab.envs.locomotion.common.rewards`
