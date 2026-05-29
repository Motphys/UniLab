# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm. The registered
env is `Go2ArmManipLoco`, the PPO owner is
`conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`, and the HIM-PPO owner is
`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## PPO

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

## HIM-PPO

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

The env currently raises if constructed with a backend other than MuJoCo. Keep
backend selection in `task=go2_arm_manip_loco/mujoco`, and do not override
`training.sim_backend` alone.

See {doc}`../tasks/manip_loco` for the task entry.
