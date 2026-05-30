# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm. The registered
env is `Go2ArmManipLoco`, the PPO owner is
`conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`, and the HIM-PPO owner is
`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

HIM-PPO is configured by `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` and
implemented by `scripts/train_him_ppo.py`. It is not currently declared as a
top-level `uv run train --algo ...` route in `src/unilab/cli.py`.

The env currently raises if constructed with a backend other than MuJoCo. Keep
backend selection in `--task go2_arm_manip_loco --sim mujoco`, and do not
override `training.sim_backend` alone.

See {doc}`../4-tasks/4-manip_loco` for the task entry.
