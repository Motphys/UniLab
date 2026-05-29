# Task Owner Config Contract

Task owner YAML is the identity of a composed task/backend/algorithm path. The
contract is recorded in
{doc}`/adr/ADR-0003-task-owner-and-config-compose-contract`.

## Owner Paths

- PPO and APPO owner YAMLs use
  `conf/{ppo,appo}/task/<task>/<backend>.yaml`.
- MLX PPO composes from `conf/ppo/config_mlx.yaml` and reuses the PPO task
  owner YAML layout.
- Off-policy owner YAMLs include the algorithm dimension:
  `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`.
- Other existing config roots, such as `conf/ppo_him/` and
  `conf/hora_distill/`, follow the same owner-YAML identity rule for their
  supported tasks.

## Required Semantics

- Use the Hydra task choice to switch backend, for example
  `task=go2_joystick_flat/mujoco` or `task=go2_joystick_flat/motrix`.
- For off-policy entrypoints, keep `algo=<algo>` and
  `task=<algo>/<task>/<backend>` aligned.
- `training.sim_backend` is an identity field inside the selected owner YAML. It
  is not an independent backend switch.
- Backend-specific reward, env, scene, and algorithm differences belong in the
  owner YAML, not in training scripts.
- Reward config must be explicitly injected by the owner YAML when the task uses
  rewards.

## Evidence In Repo

- PPO owner example: `conf/ppo/task/go2_joystick_flat/mujoco.yaml`
- APPO config root: `conf/appo/config.yaml`
- Off-policy config root: `conf/offpolicy/config.yaml`
- Off-policy task/algo guard: `src/unilab/training/common.py`
- Config tests: `tests/config/test_config_system.py`,
  `tests/scripts/test_train_script_configs.py`,
  `tests/envs/locomotion/g1/test_issue175_regression.py`
