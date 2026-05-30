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
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

For off-policy, `--algo` selects the first owner-path segment under
`conf/offpolicy/task/<algo>/`; do not include the algorithm name in `--task`.

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
is not an independent backend switch. Use the unified CLI `--sim` flag to
select the backend.

See the developer contract in {doc}`../../4-developer_guide/2-contracts/3-task_owner`.
