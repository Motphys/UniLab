# Manip-Loco

语言: 简体中文

`go2_arm_manip_loco` 将 Go2 运动与 Airbot 机械臂结合。已注册的 env 是 `Go2ArmManipLoco`，PPO owner 是 `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`，HIM-PPO owner 是 `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

HIM-PPO 由 `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` 配置，由 `scripts/train_him_ppo.py` 实现。它目前没有在 `src/unilab/cli.py` 中声明为顶层 `uv run train --algo ...` 路由。

如果用 MuJoCo 以外的后端构造该 env，它目前会抛出异常。请将后端选择保持在 `--task go2_arm_manip_loco --sim mujoco`，不要单独覆盖 `training.sim_backend`。

关于任务入口，参见 {doc}`../4-tasks/4-manip_loco`。

## Navigation

- Index: [文档](0-index.md)
