# 从 UniLab 1.4 迁移到 1.5

UniLab 1.5 将公共 reset 契约恢复为 IsaacLab/mjlab 风格的 `env_ids` 行索引形态（
[issue #2460](https://github.com/Motphys/UniLab/issues/2460)，隶属 roadmap
[#2459](https://github.com/Motphys/UniLab/issues/2459)）。v1.4 的定宽 bool mask
形态已直接移除、无兼容层
（[issue #2474](https://github.com/Motphys/UniLab/issues/2474)）：向公共 reset
边界传 bool mask 会抛 `TypeError`。

本页说明签名变更、已移除的 mask 形态、mask-preserving term 如何回退为 `env_ids`
实现，以及下游任务如何删除 v1.4 时期的 temporal workaround。1.4 之前的实现即为恢复后
形态的参考——例如 `git show v1.3.4:src/unilab/managers/event_manager.py`。

## 变更清单

| 接口 | v1.4 | v1.5 |
| --- | --- | --- |
| `TorchEnv.reset`、`ManagerBasedRlEnv.reset` | `reset(reset_mask: Tensor \| None = None)` | `reset(env_ids: Tensor \| None = None)` |
| `ManagerBase.reset`（reward、observation、termination、curriculum、action、command、metrics、event 各 manager） | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor \| slice \| None = None)` |
| 有状态 class term 的 reset 钩子（reward/observation/termination/curriculum/metric） | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor)` |
| action term | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor \| None = None)` |
| `mode="reset"` 的 event term | 必须实现 class 边界 `reset(env, reset_mask, **params)` | `func(env, env_ids, **params)`；普通函数重新可用 |
| `EventManager.apply` | `apply(mode="reset")` 抛 `ValueError` | `apply(mode="reset", env_ids=..., global_env_step_count=...)` 负责派发 reset 事件 |
| `EventManager.dispatch_reset` | 唯一的 reset 事件入口 | 已移除；使用 `apply(mode="reset", env_ids=...)` |
| `EventTermCfg.min_step_count_between_reset` | reset 事件不生效 | 恢复逐行触发过滤（1.3.x 语义） |
| reset owner | `reset_transaction(reset_mask)` / `reset_committed(reset_mask)` | `reset_transaction(env_ids)` / `reset_committed(env_ids)` |
| recorder 钩子 | `record_pre_reset(reset_mask)` / `record_post_reset(reset_mask)` | `record_pre_reset(env_ids)` / `record_post_reset(env_ids)` |
| RL 环境契约 | `EnvProtocol.reset(env_indices)` | `EnvProtocol.reset(env_ids)`（见 `src/unilab/base/contracts/env.py`） |

`env_ids` 是一维整数行索引张量；`None`（manager 边界还接受 `slice`）表示全部行。
恢复后的形态与 v1.3.x 公共契约一致，针对 1.3.x 编写的代码无需修改。

公共 `env_ids` 形态与 backend 事务所消费的定宽 bool mask 之间的内部转换只有唯一
一处：`src/unilab/base/reset_selectors.py`
（`resolve_reset_env_ids`、`reset_env_ids_mask`、`resolve_reset_mask`）。
backend 不受影响——见下方 FAQ。

## 已移除的 mask 形态

- **v1.5** —— 公共契约使用 `env_ids`。**位置传参**的定宽 bool 张量会抛
  `TypeError`。仍定义 v1.4 边界 `reset(env, reset_mask, **params)` 的
  class 型 reset 事件项不再被检测：reset term 只以 `func(env, env_ids, **params)`
  形式被调用。`EventManager.dispatch_reset` 已移除；请使用
  `apply(mode="reset", env_ids=...)`。

关键字形式同样报错：v1.5 中 `env.reset(reset_mask=mask)` 会立即抛
`TypeError`。请将参数名改为 `env_ids` 并传行索引。

## 将 mask-preserving term 回退为 `env_ids` 实现

机械规则：

- 将 mask 参数替换为一维 int64 的 `env_ids` 张量；
- `values.masked_fill_(mask, x)` 改为 `values[env_ids] = x`；
- `torch.where(mask[:, None], new, old)` 改为 `values[env_ids] = new_rows`；
- term 内部如需由 ids 得到 mask，使用 `unilab.base.reset_selectors` 的
  `reset_env_ids_mask(env_ids, num_envs=env.num_envs, device=env.device)`，
  不要在别处手写转换；
- 采样数量用 `env_ids.numel()`，而不是 `int(mask.sum())`。

### Event term（`mode="reset"`）

v1.4 要求每个 reset 事件项实现 mask-preserving 的 class 边界。v1.5 以 mjlab 风格
调用 reset term，普通函数重新可用：

```python
# v1.4 —— mask-preserving class term
class randomize_link_friction:
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        self._friction = torch.ones(env.num_envs, device=env.device)

    def reset(
        self,
        env: ManagerBasedRlEnv,
        reset_mask: torch.Tensor,
        friction_range: tuple[float, float],
    ) -> None:
        rows = torch.nonzero(reset_mask, as_tuple=False).flatten()
        sampled = env.rng.uniform(*friction_range, rows.numel())
        self._friction[reset_mask] = sampled
```

```python
# v1.5 —— term 以 func(env, env_ids, **params) 形式被调用；
# class term 将边界移到 __call__（普通函数亦可）
class randomize_link_friction:
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        self._friction = torch.ones(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        friction_range: tuple[float, float],
    ) -> None:
        sampled = env.rng.uniform(*friction_range, env_ids.numel())
        self._friction[env_ids] = sampled
```

有状态 class term 注意区分两个钩子：`func(env, env_ids, **params)` 是 reset 事件
本体，而 class 上单参数的 `reset(self, env_ids)` 仍是状态清理钩子（参见
`src/unilab/envs/mdp/events.py` 中 `apply_body_impulse.reset`）。双参数的
`reset(env, reset_mask, **params)` 形态已移除，不再被派发。写仿真状态的 event
term 仍通过环境持有的 packed reset transaction 提交，只有行选择器形态变化。

### Reward / termination / curriculum / metric 的 class term

```python
# v1.4
class episode_length_reward:
    def reset(self, reset_mask: torch.Tensor) -> None:
        self._steps.masked_fill_(reset_mask, 0)
```

```python
# v1.5
class episode_length_reward:
    def reset(self, env_ids: torch.Tensor) -> None:
        self._steps[env_ids] = 0
```

### Observation class term

钩子变化与 reward term 相同（参见
`src/unilab/envs/mdp/observations.py` 中 `_NamedSensorObservation.reset`）。
自定义 noise model 是例外：
`src/unilab/managers/_noise/noise_model.py` 中
`NoiseModel.reset(reset_mask)` 保持 mask 形态——`ObservationManager` 会替你完成
`env_ids` 到 mask 的转换，自定义 noise model 无需修改。

### Action term

```python
# v1.4
def reset(self, reset_mask: torch.Tensor) -> None:
    self._raw_actions.masked_fill_(reset_mask.reshape(-1, 1), 0.0)
```

```python
# v1.5（与 src/unilab/envs/mdp/actions/actions.py 一致）
def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
        env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
```

### Command term

无需修改。公共 `CommandTerm.reset(env_ids=None, ...)` 在 manager 边界完成 mask
转换；受保护的 `_resample_command(reset_mask)` 子类钩子有意保持 v1.4 的定宽 mask
形态。

### 自定义 `TorchEnv` 子类

重命名行选择器，并在 backend 事务边界只做一次转换：

```python
from unilab.base.reset_selectors import reset_env_ids_mask, resolve_reset_env_ids


def reset(self, env_ids: torch.Tensor | None = None):
    ids = resolve_reset_env_ids(
        env_ids, num_envs=self.num_envs, device=self.device, owner="MyEnv.reset"
    )
    ...
    backend_mask = reset_env_ids_mask(ids, num_envs=self.num_envs, device=self.device)
```

`resolve_reset_env_ids` 直接实施 v1.5 契约：bool mask 抛 `TypeError`、
越界索引同样被拒绝，因此 env 实现可以直接获得契约行为。

## 删除任务层 temporal workaround

v1.4 因缺少 mask reset 边界而移除了 manager 持有的 temporal 状态，任务只能自行堆叠
history。以 Wuji reorient 任务为例，它实现了任务级的 `HistoryFeature`（自带
`CircularBuffer`、通过 `reset(reset_mask)` 钩子做 `reset_mask.nonzero()` 解包，
另有 `NoisyHistoryFeature` 子类在每帧入栈前用 `env.rng` 手动加噪）。

随着 `ObservationTermCfg` 恢复 `history_length` 与 `delay_*` 原生配置（issue
[#2462](https://github.com/Motphys/UniLab/issues/2462)；语义与 1.3.4 一致——按
**oldest → newest** 时间顺序排布，reset 行用 reset 后首帧**回填**（backfill），
每个 term 的处理流水线为
`compute → noise → clip → scale → delay → history`，即噪声在入 history
**之前**施加），任务层的这些类可以删除：

```yaml
# v1.4 任务 YAML（任务自持 history + 手写加噪）
joint:
  func: wuji_unilab.tasks.reorient.observations.NoisyHistoryFeature
  params:
    kind: joint
    history_length: 3
    low: -0.06
    high: 0.06
```

```yaml
# v1.5 —— 普通 term + 原生 history_length 与 noise 配置
joint:
  func: wuji_unilab.tasks.reorient.observations.feature
  params: {kind: joint}
  history_length: 3
  noise:
    _target_: unilab.managers._noise.UniformNoiseCfg
    n_min: -0.06
    n_max: 0.06
    operation: add
```

展平后的宽度不变（`history_length × term_dim`，term-major，最旧在前——与
`HistoryFeature` 发布的布局一致），因此已训练策略的输入维度可以直接沿用。噪声只在
`enable_corruption: true` 的组上生效，与 workaround 的 actor-only 加噪一致。额外
声明 packed-read 元数据（`tensor_state_entity` / `tensor_body_names`，例如 Wuji 的
`TipsFeature`）的 class term 可以继续保持 class 形态：保留 `__call__` 与元数据
property，删除 buffer、`reset` 钩子与手写加噪，并把
`history_length`/`scale`/`noise` 移到 term 配置上。

延迟观测改用 `ObservationTermCfg` 的原生 `delay_min_lag` / `delay_max_lag` /
`delay_per_env` / `delay_hold_prob` / `delay_update_period` / `delay_per_env_phase`
字段，不再使用任务层延迟队列；当组内所有 term 共享同一深度时，也可用组级
`ObservationGroupCfg.history_length` 统一覆盖。

## FAQ

**为什么 `env.reset(reset_mask=mask)` 直接抛 `TypeError` 而不是告警？**
参数本身已从 `reset_mask` 更名为 `env_ids`，且 v1.4 的 mask 形态已不带
兼容窗口地移除：关键字形式与位置传参的 bool 张量在 v1.5 都会立即抛
`TypeError`。立即报错可以强迫用户现在就完成这一处改名，而不是在以后的
版本中爆发。

**自定义 backend 或自定义 reset 事务需要修改吗？** 不需要。
`SimBackend.apply_reset` 与 packed reset 事务仍消费定宽 bool mask；只有公共
用户边界恢复为 `env_ids`。`src/unilab/base/reset_selectors.py` 是唯一的转换点。

**自定义 command term 或 noise model 需要修改吗？** 不需要。command term 覆盖的
受保护钩子 `_resample_command(reset_mask)` 保持 v1.4 的 mask 形态；自定义
`NoiseModel` 子类保持 `reset(reset_mask)`。两个 manager 都在公共边界完成转换。

**如何迁移 `EventManager.dispatch_reset(mask, ...)` 调用？**
`dispatch_reset` 已在 v1.5 移除。替换为
`event_manager.apply(mode="reset", env_ids=mask.nonzero(as_tuple=False).flatten(), global_env_step_count=...)`；
如果你手上已有行索引，直接传入即可。

**v1.4 中 `min_step_count_between_reset` 的过滤去哪了？** v1.4 的
`dispatch_reset` 忽略了该配置；v1.5 在
`EventManager.apply(mode="reset", ...)` 中恢复了 1.3.x 的逐行过滤，reset 事件
重新遵守两次触发之间的最小步数间隔。
