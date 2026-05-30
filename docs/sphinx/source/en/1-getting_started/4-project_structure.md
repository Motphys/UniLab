# Project Structure

UniLab keeps runtime contracts, configuration, training scripts, and docs in
separate owner areas. Use this map when you need to find the right layer before
changing behavior.

| Path | Owner Role |
| --- | --- |
| `scripts/` | Thin training and tooling entrypoints. Scripts compose Hydra config and call owner-layer code. |
| `conf/` | Hydra roots and task owner YAMLs. The top-level CLI exposes backend selection as `--task` plus `--sim`, then composes the matching owner YAML. |
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
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Do not switch backends by overriding `training.sim_backend` alone.

## Where To Go Next

- User training commands: {doc}`../2-user_guide/1-training/1-cli_reference`
- Hydra owner YAMLs: {doc}`../2-user_guide/1-training/2-hydra_config`
- Contracts for contributors: {doc}`../4-developer_guide/0-index`
