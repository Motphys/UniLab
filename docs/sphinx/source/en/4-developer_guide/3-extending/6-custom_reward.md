# Walkthrough: Add a Custom Reward Term

This walkthrough adds a reward term from zero to the point where
`reward/<term>` shows up in the training log. The runtime is Torch-only (since
1.3.4; see the changelog and ADR-0011) — terms return `torch.Tensor`, and there
is no NumPy path.

## 1. Define the Term

Task-owned terms live in the task package, e.g.
`src/unilab/tasks/locomotion/common/sensor_reward_terms.py`; framework-level
terms live in `src/unilab/envs/mdp/rewards.py` and are re-exported at
`unilab.envs.mdp.<name>`. Put a new task-specific term in the owning task
package.

A term is a plain function taking the env first, then its `params`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


def action_magnitude_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sum of absolute raw policy actions, one value per environment."""
    return torch.sum(torch.abs(env.action_manager.action), dim=1)
```

Return contract, enforced by `RewardManager._compute_term`
(`src/unilab/managers/reward_manager.py`): a `torch.Tensor` of shape
`(num_envs,)`, dtype `float32`, on the env device. Wrong type, dtype, device,
or shape raises at the first `compute()` call; a non-finite value raises with
the term name. The manager invokes the term as `func(env, **params)`, so every
key under `params:` in the YAML must be an accepted parameter of the function.

For stateful or sensor-bound terms use the class form instead: the class is
instantiated once with `(cfg, env)` at manager construction and then called per
step as `func(env, **params)`; an optional `reset(env_ids)` clears per-episode
state (`ManagerTermBaseCfg` in `src/unilab/managers/manager_base.py`). Binding a
named XML sensor at construction is the `SensorTermBase` pattern in
`src/unilab/tasks/locomotion/common/manager_terms.py`.

## 2. Wire It into the Owner YAML

Reward terms live under the top-level `reward:` mapping of the task owner YAML
(`src/unilab/conf/<algo>/task/<task>/base.yaml` or a backend sibling), one
block per term with `func`, `weight`, and optional `params` — real structure
from `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`:

```yaml
reward:
  action_magnitude:
    func: unilab.tasks.locomotion.common.manager_terms.action_magnitude_l1
    weight: -0.001
```

`func` is an importable `module.path.name`. The step reward contributed by the
term is `raw_value * weight` (scaled by `dt` inside the env; the log shows the
pre-dt weighted rate). Use a negative weight for penalties — the framework
penalty terms return non-negative costs, so do not negate them a second time.

Verify composition (still passes the CLI's backend preflight; `--sim mujoco`
needs the `mujoco` extra):

```bash
uv run train --algo ppo --task <task> --sim mujoco --cfg job \
  | grep -A3 'action_magnitude'
# Expected: the func/weight lines from your owner YAML.
```

## 3. See It in the Training Log

Run a short training session (needs the backend's extra; a GPU for any
meaningful run):

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.no_play=true
```

Manager-Based environments publish per-term reward diagnostics as
`reward/<term>` keys in `info["log"]` — produced by
`RewardManager.step_reward_extras()`
(`src/unilab/managers/reward_manager.py`, mean across environments of
`raw_value * weight`, before `dt` scaling) — and the training logger forwards
them to TensorBoard / W&B ({doc}`../../2-user_guide/1-training/3-logging`):

```bash
uv run tensorboard --logdir logs/rsl_rl_ppo/<task>
# Expected: a reward/action_magnitude tag alongside the other reward/<term> tags.
```

If the term misbehaves, the failure is explicit: construction errors name the
term, a wrong return type/shape raises from `RewardManager`, and a NaN/Inf
return raises `RewardManager term '<term>' returned NaN/Inf` (plus a NaN-guard
state dump when enabled). Diagnosis workflow:
{doc}`../../2-user_guide/1-training/8-training_diagnostics`.

## 4. Tune the Weight

Retune a declared term from the command line — no YAML edit needed:

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  reward.action_magnitude.weight=-0.01
```

This works because `reward` is a root-level Hydra section of the composed
config (see the "Reward term weights" section of
{doc}`../../2-user_guide/2-algorithms/1-ppo`). Add new terms in the owner YAML
itself so the task stays reproducible; the CLI override is for sweeps.

When porting terms from another framework, check the built-in tables first —
the term may already exist: {doc}`../../5-reference/6-manager_terms/1-rewards`.
The Legged Gym mapping guide is
{doc}`../../3-deployment/3-framework_migration/6-reward_porting`.
