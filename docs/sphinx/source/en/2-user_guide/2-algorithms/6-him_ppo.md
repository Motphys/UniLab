# HIM-PPO

HIM-PPO has its own config group and script. The entrypoint is
`scripts/train_him_ppo.py`, the base config is `conf/ppo_him/config.yaml`, and
the committed task owner is `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## Current Entrypoint

`src/unilab/cli.py` currently exposes `1-ppo`, `8-mlx_ppo`, `2-appo`, `3-sac`, `4-td3`,
and `flashsac` through the top-level `uv run train` CLI. HIM-PPO is implemented
by `scripts/train_him_ppo.py`, but it does not yet have a top-level `--algo`
route.

## Owner Details

The Go2 arm owner fills the required history dimensions from the base config:

- `algo.num_one_step_obs=76`
- `algo.num_actor_history=5`
- `algo.num_critic_history=1`
- `training.task_name=Go2ArmManipLoco`

Playback uses the same HIM-PPO implementation entrypoint once a checkpoint is
available. Keep user-facing PPO examples on the supported top-level CLI shape;
use the HIM-PPO script path only when debugging that specialized stack.

HIM-PPO is not the default PPO path; use it for the Go2 arm manip-loco owner
that explicitly selects the HIM-PPO config group.
