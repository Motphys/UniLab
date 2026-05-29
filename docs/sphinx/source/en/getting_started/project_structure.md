# Project Structure

UniLab keeps runtime contracts, configuration, training scripts, and docs in
separate owner areas. Use this map when you need to find the right layer before
changing behavior.

| Path | Owner Role |
| --- | --- |
| `scripts/` | Thin training and tooling entrypoints. Scripts compose Hydra config and call owner-layer code. |
| `conf/` | Hydra roots and task owner YAMLs. Backend selection lives in task owner paths such as `task=go2_joystick_flat/mujoco`. |
| `src/unilab/base/` | Registry, env state, scene, and backend contracts. |
| `src/unilab/envs/` | Task env implementations and task-specific reset, reward, observation, and DR logic. |
| `src/unilab/algos/` | PPO, APPO, off-policy, MLX, HIM-PPO, and HORA algorithm code. |
| `src/unilab/ipc/` | Shared-memory and async runner primitives. |
| `src/unilab/training/` | Shared training helpers for logging, playback, seed handling, and config guards. |
| `src/unilab/visualization/` | Playback, rendering, NaN inspection, and scene/export utilities. |
| `tests/` | Contract, config, env, algorithm, script, and integration tests. |
| `docs/sphinx/source/en/` | English user, deployment, developer, and reference docs. |
| `docs/sphinx/source/zh_CN/` | Chinese docs with compatibility paths handled by the language switcher. |

## Config Layout

The main config roots are:

- `conf/ppo/config.yaml` for torch PPO.
- `conf/ppo/config_mlx.yaml` for MLX PPO.
- `conf/appo/config.yaml` for APPO.
- `conf/offpolicy/config.yaml` plus `conf/offpolicy/algo/*.yaml` for SAC,
  TD3, and FlashSAC.
- `conf/ppo_him/config.yaml` and `conf/hora_distill/config.yaml` for the
  specialized HIM-PPO and HORA paths.

Task owner YAMLs are the backend identity. Examples:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

Do not switch backends by overriding `training.sim_backend` alone.

## Where To Go Next

- User training commands: {doc}`../user_guide/training/cli_reference`
- Hydra owner YAMLs: {doc}`../user_guide/training/hydra_config`
- Contracts for contributors: {doc}`../developer_guide/index`
