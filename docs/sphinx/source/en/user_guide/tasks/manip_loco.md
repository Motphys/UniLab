# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm. The registered
env is `Go2ArmManipLoco`.

## Owner Configs

- PPO owner: `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`
- HIM-PPO owner: `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`
- Scene entry: `src/unilab/assets/robots/go2_arm/scene_flat.xml`

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

## HIM-PPO

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

The current committed owner path is MuJoCo. Keep backend selection in
`task=go2_arm_manip_loco/mujoco`, and do not override
`training.sim_backend` alone.
