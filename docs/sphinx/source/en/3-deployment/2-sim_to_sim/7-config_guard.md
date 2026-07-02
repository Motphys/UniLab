# Cross-Backend Config Guard

When replaying across backends (train on one backend, `eval` on another), UniLab automatically checks whether the target backend config is compatible with the policy contract captured at training time, so a mismatched config can't silently load a broken policy. This happens with no manual steps.

## An example that replays successfully

For `go2_joystick_flat`, the MuJoCo and Motrix owners agree on every guarded field, so a cross-backend replay passes directly:

```bash
# 1) Train in MuJoCo, producing a checkpoint
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# 2) Replay the same checkpoint across backends in Motrix — the guard passes, playback runs
uv run eval  --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
```

## How the guard chain works

1. **At train time**: `ExperimentTracker` snapshots the contract fields that define policy I/O into `contract_snapshot` in `run_config.json` (the checkpoint format is untouched, so historical checkpoints stay compatible).
2. **At replay time**: `eval` loads the **target backend** owner config selected by `--sim` (e.g. `conf/ppo/task/go2_joystick_flat/motrix.yaml`) and injects `training.play_only=true`.
3. **Before env creation**: the five play entrypoints (rsl_rl / appo / offpolicy / mlx_ppo / him_ppo) call `resolve_sim2sim_config`, comparing the target config against the source run's contract snapshot field by field.
4. **At weight load**: `policy_load_dim_guard` wraps checkpoint loading, re-raising cryptic tensor shape-mismatch errors as a clear sim2sim diagnostic.

## What the guard covers

Fields are classified by dotted path into three tiers (see `src/unilab/training/sim2sim.py`):

| Tier | Behavior | Fields |
|---|---|---|
| **DENYLIST** | Mismatch → `CrossBackendIncompatibleError`, aborts | `algo.obs_groups`, `env.control_config.action_scale`, `algo.policy.actor_hidden_dims` / `critic_hidden_dims`, `algo.empirical_normalization` / `algo.obs_normalization`, `env.sampling_mode` |
| **WARNING_LIST** | Prints a warning, continues | `reward.*`, `env.control_config.simulate_action_latency`, `env.ctrl_dt` |
| **ALLOWLIST** | Free to override, not checked | `training.sim_backend`, `env.scene`, `training.play_steps`, `env.domain_rand`, `env.noise_config`, `env.commands.vel_limit` |

## When DENYLIST fields differ

If the target backend's DENYLIST fields differ from training (e.g. a task whose two owners use different `action_scale` values), the guard aborts before env creation and lists the diverging fields. Two ways to resolve:

- **Align the contract** (recommended): make the target owner's DENYLIST fields match the training backend, then replay.
- **Force through** (at your own risk): `uv run eval ... training.sim2sim_strict=false` downgrades DENYLIST mismatches to warnings.

> Legacy runs: if `run_config.json` has no `contract_snapshot` (older training), the guard skips with a warning instead of breaking your workflow.

## See also

- {doc}`1-backend_swap`
- {doc}`4-reward_parity`
- {doc}`../../4-developer_guide/9-sim2sim_contract_status`
