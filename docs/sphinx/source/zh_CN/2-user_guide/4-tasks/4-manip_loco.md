# Manip-Loco

`go2_arm_manip_loco` 将 Go2 运动控制与 Airbot 机械臂结合。其注册的
env 是 `Go2ArmManipLoco`。

## Owner Configs

- PPO owner：`conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`
- HIM-PPO owner：`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`
- 场景入口：`src/unilab/assets/robots/go2_arm/scene_flat.xml`

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

HIM-PPO owner 是 `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。
`src/unilab/cli.py` 当前未将 HIM-PPO 作为顶层
`uv run train --algo ...` 路线暴露。

当前已提交的 owner 路径是 MuJoCo。后端选择请保留在
`--task go2_arm_manip_loco --sim mujoco` 中，不要单独 override
`training.sim_backend`。
