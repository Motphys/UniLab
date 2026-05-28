# HIM-PPO

HIM-PPO has its own config group and script. The entrypoint is
`scripts/train_him_ppo.py`, the base config is `conf/ppo_him/config.yaml`, and
the committed task owner is `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## Quick Start

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

## Owner Details

The Go2 arm owner fills the required history dimensions from the base config:

- `algo.num_one_step_obs=76`
- `algo.num_actor_history=5`
- `algo.num_critic_history=1`
- `training.task_name=Go2ArmManipLoco`

Playback uses the same script:

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

HIM-PPO is not the default PPO path; use it for the Go2 arm manip-loco owner
that explicitly selects the HIM-PPO config group.
