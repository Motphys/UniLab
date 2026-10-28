# Migrating from UniLab 1.4 to 1.5

UniLab 1.5 restores the IsaacLab/mjlab-style `env_ids` row-index form as the
public reset contract ([issue
#2460](https://github.com/Motphys/UniLab/issues/2460), part of roadmap
[#2459](https://github.com/Motphys/UniLab/issues/2459)). The v1.4 fixed-width
bool mask form is removed without a compatibility layer ([issue
#2474](https://github.com/Motphys/UniLab/issues/2474)): passing a bool mask to
a public reset boundary raises `TypeError`.

This page covers the signature changes, the removed mask forms, how to port
mask-preserving terms back to `env_ids`, and how downstream tasks drop their
v1.4 temporal workarounds. The pre-1.4 implementations are the reference for
the restored form — for example
`git show v1.3.4:src/unilab/managers/event_manager.py`.

## What changed

| Surface | v1.4 | v1.5 |
| --- | --- | --- |
| `TorchEnv.reset`, `ManagerBasedRlEnv.reset` | `reset(reset_mask: Tensor \| None = None)` | `reset(env_ids: Tensor \| None = None)` |
| `ManagerBase.reset` (reward, observation, termination, curriculum, action, command, metrics, event managers) | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor \| slice \| None = None)` |
| Class term reset hooks (stateful reward/observation/termination/curriculum/metric terms) | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor)` |
| Action terms | `reset(reset_mask: Tensor)` | `reset(env_ids: Tensor \| None = None)` |
| Event terms with `mode="reset"` | class boundary `reset(env, reset_mask, **params)` (required) | `func(env, env_ids, **params)`; plain functions work again |
| `EventManager.apply` | `apply(mode="reset")` raised `ValueError` | `apply(mode="reset", env_ids=..., global_env_step_count=...)` dispatches reset terms |
| `EventManager.dispatch_reset` | the only reset-event entry point | removed; use `apply(mode="reset", env_ids=...)` |
| `EventTermCfg.min_step_count_between_reset` | ignored for reset events | per-row trigger filtering restored (1.3.x semantics) |
| Reset owners | `reset_transaction(reset_mask)` / `reset_committed(reset_mask)` | `reset_transaction(env_ids)` / `reset_committed(env_ids)` |
| Recorder hooks | `record_pre_reset(reset_mask)` / `record_post_reset(reset_mask)` | `record_pre_reset(env_ids)` / `record_post_reset(env_ids)` |
| RL env contract | `EnvProtocol.reset(env_indices)` | `EnvProtocol.reset(env_ids)` (see `src/unilab/base/contracts/env.py`) |

`env_ids` is a one-dimensional integer tensor of row indices; `None` (and, at
manager boundaries, a `slice`) selects every row. The restored form matches
the v1.3.x public contract, so code written against 1.3.x works unchanged.

The internal conversion between the public `env_ids` form and the fixed-width
bool masks that backend transactions consume lives in exactly one place:
`src/unilab/base/reset_selectors.py`
(`resolve_reset_env_ids`, `reset_env_ids_mask`, `resolve_reset_mask`).
Backends are unaffected — see the FAQ below.

## Removed mask forms

- **v1.5** — the public contract speaks `env_ids`. Passing a fixed-width bool
  tensor **positionally** to `reset(...)` raises `TypeError`. A class-based
  reset event term that still defines the v1.4 boundary
  `reset(env, reset_mask, **params)` is no longer detected: reset terms are
  called as `func(env, env_ids, **params)` only. `EventManager.dispatch_reset`
  is removed; use `apply(mode="reset", env_ids=...)`.

The keyword form fails the same way: `env.reset(reset_mask=mask)` raises
`TypeError` immediately in v1.5. Rename the argument to `env_ids` and pass row
indices.

## Porting mask-preserving terms to `env_ids`

The mechanical rules are:

- replace the mask parameter with a 1-D int64 `env_ids` tensor;
- `values.masked_fill_(mask, x)` becomes `values[env_ids] = x`;
- `torch.where(mask[:, None], new, old)` becomes `values[env_ids] = new_rows`;
- to derive a mask from ids inside a term, use
  `reset_env_ids_mask(env_ids, num_envs=env.num_envs, device=env.device)` from
  `unilab.base.reset_selectors` — do not hand-roll conversions elsewhere;
- sample `env_ids.numel()` rows instead of `int(mask.sum())`.

### Event terms (`mode="reset"`)

v1.4 required every reset event term to expose a mask-preserving class
boundary. v1.5 calls reset terms mjlab-style, so a plain function works again:

```python
# v1.4 — mask-preserving class term
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
# v1.5 — the term is called as func(env, env_ids, **params); a class term
# moves the boundary onto __call__ (a plain function works too)
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

For stateful class terms, note the two distinct hooks: `func(env, env_ids,
**params)` is the reset event itself, while a one-argument
`reset(self, env_ids)` on the class remains the state-clearing hook (see
`apply_body_impulse.reset` in `src/unilab/envs/mdp/events.py`). The
two-argument `reset(env, reset_mask, **params)` form is removed and no longer
dispatched. Event terms that write simulator state keep staging through the
environment-owned packed reset transaction; only the row selector changes.

### Reward, termination, curriculum, and metric class terms

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

### Observation class terms

Same hook change as reward terms (see `_NamedSensorObservation.reset` in
`src/unilab/envs/mdp/observations.py`). Custom noise models are the exception:
`NoiseModel.reset(reset_mask)` in `src/unilab/managers/_noise/noise_model.py`
keeps the mask form — the `ObservationManager` converts `env_ids` to a mask
for you, so custom noise models need no change.

### Action terms

```python
# v1.4
def reset(self, reset_mask: torch.Tensor) -> None:
    self._raw_actions.masked_fill_(reset_mask.reshape(-1, 1), 0.0)
```

```python
# v1.5 (matches src/unilab/envs/mdp/actions/actions.py)
def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
        env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
```

### Command terms

No change needed. The public `CommandTerm.reset(env_ids=None, ...)` converts
to a mask at the manager boundary; the protected `_resample_command(reset_mask)`
subclass hook intentionally keeps the fixed-width mask form from v1.4.

### Custom `TorchEnv` subclasses

Rename the selector and convert once, at the backend transaction edge:

```python
from unilab.base.reset_selectors import reset_env_ids_mask, resolve_reset_env_ids


def reset(self, env_ids: torch.Tensor | None = None):
    ids = resolve_reset_env_ids(
        env_ids, num_envs=self.num_envs, device=self.device, owner="MyEnv.reset"
    )
    ...
    backend_mask = reset_env_ids_mask(ids, num_envs=self.num_envs, device=self.device)
```

`resolve_reset_env_ids` enforces the v1.5 contract for you: it rejects bool
masks with `TypeError` and rejects out-of-range indices, so env
implementations get the contract behavior for free.

## Dropping task-level temporal workarounds

v1.4 removed manager-owned temporal state because it had no mask reset
boundary, so tasks stacked history themselves. The Wuji reorient task, for
example, carries a task-owned `HistoryFeature` class (own `CircularBuffer`,
a `reset(reset_mask)` hook that unpacks `reset_mask.nonzero()`, and a
`NoisyHistoryFeature` subclass that draws noise through `env.rng` before each
frame enters the stack).

With history and delay restored on `ObservationTermCfg` (issue
[#2462](https://github.com/Motphys/UniLab/issues/2462); semantics identical to
1.3.4 — chronological **oldest → newest** layout, reset rows **backfilled**
with the first post-reset frame, and the per-term pipeline
`compute → noise → clip → scale → delay → history`, so noise enters **before**
history), the task-level classes can be deleted:

```yaml
# v1.4 task YAML (task-owned history + hand-rolled noise)
joint:
  func: wuji_unilab.tasks.reorient.observations.NoisyHistoryFeature
  params:
    kind: joint
    history_length: 3
    low: -0.06
    high: 0.06
```

```yaml
# v1.5 — plain term + native history_length and noise config
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

The flattened width is unchanged (`history_length × term_dim`, term-major,
oldest first — the same layout `HistoryFeature` published), so trained policy
shapes carry over. Noise only applies to groups with
`enable_corruption: true`, which matches the workaround's actor-only noise.
Class terms that additionally declare packed-read metadata
(`tensor_state_entity` / `tensor_body_names`, e.g. Wuji's `TipsFeature`) can
stay class-based: keep `__call__` and the metadata properties, delete the
buffer, the `reset` hook, and the hand-rolled noise, and move
`history_length`/`scale`/`noise` onto the term config.

For delayed observations use the native `delay_min_lag` / `delay_max_lag` /
`delay_per_env` / `delay_hold_prob` / `delay_update_period` /
`delay_per_env_phase` fields on `ObservationTermCfg` instead of task-level
delay queues; a group-level `ObservationGroupCfg.history_length` overrides all
terms in a group when every term shares one depth.

## FAQ

**Why does `env.reset(reset_mask=mask)` raise `TypeError` instead of
warning?** The parameter itself is renamed to `env_ids`, and the v1.4 mask
form is removed without a deprecation window: both the keyword form and a
positional bool tensor raise `TypeError` immediately in v1.5. Failing fast
forces the one-word fix now instead of breaking harder later.

**Does a custom backend or a custom reset transaction need changes?** No.
`SimBackend.apply_reset` and the packed reset transaction still consume
fixed-width bool masks; only the public user-facing boundary moved back to
`env_ids`. `src/unilab/base/reset_selectors.py` is the sole conversion point.

**Do custom command terms or noise models need changes?** No. Command terms
override the protected `_resample_command(reset_mask)` hook, which keeps its
v1.4 mask form; custom `NoiseModel` subclasses keep `reset(reset_mask)`. Both
managers convert at their public boundary.

**How do I migrate a call to `EventManager.dispatch_reset(mask, ...)`?**
`dispatch_reset` is removed in v1.5. Replace it with
`event_manager.apply(mode="reset", env_ids=mask.nonzero(as_tuple=False).flatten(), global_env_step_count=...)`,
or keep passing ids you already have.

**Where did `min_step_count_between_reset` filtering go in v1.4?** v1.4's
`dispatch_reset` ignored it; v1.5 restores the 1.3.x per-row filtering in
`EventManager.apply(mode="reset", ...)`, so reset events again respect the
minimum step gap between triggers.
