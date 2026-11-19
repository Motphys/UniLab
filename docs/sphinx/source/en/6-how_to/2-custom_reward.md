# Walkthrough: Add a Custom Reward Term

This walkthrough adds a reward term from zero to the point where it runs in
a real training session and its per-term value is visible in the training
logs. The runtime is Torch-only (since 1.3.4; see the changelog and
ADR-0011) — terms return `torch.Tensor`, and there is no NumPy path.

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
block per term with `func`, `weight`, and optional `params` — the structure
used by the real terms in
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`. Wiring in the term
from step 1 looks like:

```yaml
reward:
  action_magnitude:
    func: unilab.tasks.locomotion.common.sensor_reward_terms.action_magnitude_l1
    weight: -0.001
```

`func` is an importable `module.path.name`. The step reward contributed by the
term is `raw_value * weight` (scaled by `dt` inside the env; the logged rate is
the pre-dt weighted value). Use a negative weight for penalties — the framework
penalty terms return non-negative costs, so do not negate them a second time.

Verify composition (still passes the CLI's backend preflight; `--sim mujoco`
needs the `mujoco` extra, and the routed training script imports the RL stack,
so the `unirl` extra is required as well):

```bash
uv run train --algo ppo --task <task> --sim mujoco --cfg job \
  | grep -A3 'action_magnitude'
# Expected: the func/weight lines from your owner YAML.
```

## 3. See It in the Training Log

Run a short training session (needs the backend's extra and the `unirl`
RL-stack extra; a GPU for any meaningful run):

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.no_play=true
```

The Manager-Based environment tracks per-term weighted reward means on device
(`RewardManager.step_reward_means` in
`src/unilab/managers/reward_manager.py` — mean across environments of
`raw_value * weight`, before `dt` scaling). How much of that reaches the
training log depends on the algorithm route:

- **PPO (rsl_rl)**: since 1.5.8 the per-term `reward/<term>` scalars are no
  longer copied into `info["log"]`; they travel as the fixed-width
  `info["reward_components"]` tensor, which only the off-policy runtimes
  consume. The console and TensorBoard
  (`uv run tensorboard --logdir logs/rsl_rl_ppo/<RegistryName>` — the
  directory level is the `training.task_name` registry name, not the
  snake_case CLI task) show the aggregate `Train/mean_reward`, and
  `Episode_Termination/<term>` counters appear once episodes complete.
- **SAC / FlashSAC / APPO**: the off-policy runtimes consume
  `reward_components` and emit the `reward/<term>` metric family (schema:
  `src/unilab/rl/logging/metric_schema.py`), on the console and in
  TensorBoard / W&B ({doc}`../2-user_guide/1-training/3-logging`).

Verified on 1.5.9 (MuJoCo backend, single RTX 4090). PPO
(`go2_joystick_flat`, `algo.num_envs=16 algo.max_iterations=1`) console
excerpt — the aggregate only:

```text
                            Learning iteration 0/1
                            Total steps: 384
                        Mean value loss: 0.0143
                    Mean surrogate loss: -0.0449
                            Mean reward: 0.02
                    Mean episode length: 1.00
```

SAC (`g1_walk_flat`, same overrides) prints the per-term table, and the same
`reward/<term>` tags land in TensorBoard:

```text
│ Rewards (iter 1)                                                       Value │
│ Reward                                              Mean 14.612 / Peak 14.61 │
│   alive                                                             +10.0000 │
│   feet phase                                                         +4.3746 │
│   penalty action rate                                                -0.0957 │
│   tracking ang vel                                                   +0.6677 │
│   tracking lin vel                                                   +0.4943 │
```

If the term misbehaves, the failure is explicit on every route: construction
errors name the term, a wrong return type/shape raises from `RewardManager`,
and a NaN/Inf return raises `RewardManager term '<term>' returned NaN/Inf`
(plus a NaN-guard state dump when enabled). Diagnosis workflow:
{doc}`3-training_diagnostics`.

## 4. Tune the Weight

Retune a declared term from the command line — no YAML edit needed:

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  reward.action_magnitude.weight=-0.01
```

This works because `reward` is a root-level Hydra section of the composed
config (see the "Reward term weights" section of
{doc}`../2-user_guide/2-algorithms/1-ppo`). Add new terms in the owner YAML
itself so the task stays reproducible; the CLI override is for sweeps.

To confirm an override landed, compose-only check it — put overrides before
`--cfg job` (a `--cfg` flag followed by overrides is rejected by Hydra's
argument parser). Verified on 1.5.9 against the Go2 owner:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  reward.action_rate.weight=-0.01 --cfg job | grep -A2 'action_rate'
```

```text
  action_rate:
    func: unilab.envs.mdp.action_rate_l2
    weight: -0.01
```

When porting terms from another framework, check the built-in tables first —
the term may already exist: {doc}`../5-reference/6-manager_terms/1-rewards`.
The Legged Gym mapping guide is
{doc}`../3-deployment/3-framework_migration/6-reward_porting`.
