# Reward 移植

语言: 简体中文

reward 项是大多数移植 bug 藏身之处。本食谱记录了常见的 reward 项及其 UniLab
惯用写法。

## 模式：线性 / 二次跟踪误差

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

注意：

- UniLab 的 reward 项在一个 `state` **批次**上运算（CPU 上的 NumPy）；没有
  逐 env 循环，也没有 `torch`。
- 返回逐 env 的标量 reward（形状为 `(n_envs,)`）。

## 模式：接触条件奖励

```python
def reward_feet_air_time(self, state):
    contact = state.foot_contact     # bool, (n_envs, n_feet)
    air_time = state.last_air_time   # float, (n_envs, n_feet)
    first_contact = contact & ~state.prev_contact
    reward = (air_time - self.cfg.air_time_threshold) * first_contact
    return reward.sum(axis=1)
```

注意：

- UniLab 的 `state` 携带了 `prev_contact`，因此你无需自己管理边沿检测。参见
  `unilab.envs.locomotion.common.rewards`。

## 模式：动作平滑惩罚

```python
def reward_action_rate(self, state):
    return -np.sum((state.action - state.prev_action) ** 2, axis=1)
```

它已经是 `unilab.envs.locomotion.common.rewards` 中的现成辅助函数。

## 模式：姿态惩罚

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

## 终止处理

UniLab 把**终止信号**与**终止惩罚**分离开。env 的 `terminations()` 返回一个
布尔掩码；reward registry 可以包含一个消费它的 `termination_penalty` 项。

```python
def reward_termination(self, state):
    return -state.termination.astype(np.float32) * self.cfg.termination_penalty
```

## 另请参阅

- {doc}`5-task_config_translation`
- `unilab.training.reward`
- `unilab.envs.locomotion.common.rewards`

## Navigation

- Index: [文档](0-index.md)
