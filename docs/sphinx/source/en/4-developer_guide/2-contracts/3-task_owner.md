# Task Owner Config Contract

Task owner YAML is the identity of a composed task/backend/algorithm path. The
contract is recorded in
{doc}`/adr/ADR-0003-task-owner-and-config-compose-contract`.

## Owner Paths

- PPO and APPO owner YAMLs use
  `src/unilab/conf/{ppo,appo}/task/<task>/<backend>.yaml`.
- Off-policy algorithms (SAC / TD3 / FlashSAC) each have their own config
  tree: `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`.
- Other existing config roots, such as `src/unilab/conf/ppo_him/` and
  `src/unilab/conf/hora_distill/`, follow the same owner-YAML identity rule for their
  supported tasks.

## Required Semantics

- Use public CLI flags to switch backend, for example
  `uv run train --algo ppo --task go2_joystick_flat --sim mujoco` or
  `uv run train --algo ppo --task go2_joystick_flat --sim motrix`.
- For off-policy entrypoints, `--algo <algo>` selects the per-algorithm config
  tree; the owner YAML path is `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`.
- `training.sim_backend` is an identity field inside the selected owner YAML. It
  is not an independent backend switch.
- Backend-specific reward, env, scene, and algorithm differences belong in the
  owner YAML, not in training scripts.
- Reward config must be explicitly injected by the owner YAML when the task uses
  rewards.

## Evidence In Repo

- PPO owner example: `src/unilab/conf/ppo/task/go2_joystick_flat/mujoco.yaml`
- APPO config root: `src/unilab/conf/appo/config.yaml`
- Off-policy config roots: `src/unilab/conf/{sac,td3,flashsac}/config.yaml`
- Config tests: `tests/config/test_config_system.py`,
  `tests/scripts/test_train_script_configs.py`,
  `tests/envs/locomotion/g1/test_g1_owner_contract.py`
