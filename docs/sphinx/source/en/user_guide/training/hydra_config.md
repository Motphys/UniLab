# Hydra Config

UniLab uses Hydra composition with task owner YAMLs. The owner YAML is the
identity of the task, backend, reward, scene, and task-specific runtime fields.

## Owner Paths

| Stack | Owner YAML Shape |
| --- | --- |
| PPO | `conf/ppo/task/<task>/<backend>.yaml` |
| MLX PPO | `conf/ppo/task/<task>/<backend>.yaml` with `conf/ppo/config_mlx.yaml` |
| APPO | `conf/appo/task/<task>/<backend>.yaml` |
| SAC / TD3 / FlashSAC | `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` |
| HIM-PPO | `conf/ppo_him/task/<task>/<backend>.yaml` |
| HORA distillation | `conf/hora_distill/task/<task>/<backend>.yaml` |

Examples:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

For off-policy, `algo=<algo>` must match the first segment of
`task=<algo>/<task>/<backend>`.

## Safe Overrides

Hydra overrides can tune fields inside the selected owner path:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=10 \
  algo.num_envs=128 \
  training.no_play=true
```

Common fields:

- `algo.max_iterations`
- `algo.num_envs`
- `algo.load_run`
- `algo.seed`
- `training.no_play`
- `training.play_only`
- `training.play_render_mode`
- `training.logger`

## Backend Identity

`training.sim_backend` is an identity field set by the selected owner YAML. It
is not an independent backend switch. Use the owner choice, or the unified CLI
`--sim` flag, to select the backend.

See the developer contract in {doc}`../../developer_guide/contracts/task_owner`.
