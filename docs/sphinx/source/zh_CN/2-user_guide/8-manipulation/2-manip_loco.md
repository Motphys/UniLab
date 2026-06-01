# Manip-Loco

`go2_arm_manip_loco` 将 Go2 运动与 Airbot 机械臂结合。已注册的 env 是 `Go2ArmManipLoco`，PPO owner 是 `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`，HIM-PPO owner 是 `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

HIM-PPO 由 `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` 配置，由 `scripts/train_him_ppo.py` 实现。它目前没有在 `src/unilab/cli.py` 中声明为顶层 `uv run train --algo ...` 路由。

如果用 MuJoCo 以外的后端构造该 env，它目前会抛出异常。请将后端选择保持在 `--task go2_arm_manip_loco --sim mujoco`，不要单独覆盖 `training.sim_backend`。

## 调参提示

- `env.control_config.arm_action_scale`：机械臂 residual action 的幅度。
- `env.goal_ee`：末端目标采样范围和轨迹时间。
- `reward.scales.tracking_lin_vel`、`reward.scales.tracking_ang_vel`、`reward.scales.stand_still`：底盘行为的权衡。
- `env.domain_rand`：质量、摩擦、推力和 PD 随机化会改变训练难度。

## 近风险检查

改动该任务后，运行 env contract 和 site-Jacobian 测试：

```bash
uv run pytest tests/envs/locomotion/go2_arm tests/base/backend/test_mujoco_site_jacobian.py
```

如果改过 XML 或 asset，至少确认 MuJoCo 能加载场景：

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2_arm/scene_flat.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

关于任务入口，参见 {doc}`../4-tasks/4-manip_loco`。
