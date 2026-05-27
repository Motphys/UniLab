# Go2 Arm Manip Loco Training Entry Guide

语言: 简体中文

本页说明 `go2_arm_manip_loco` 的训练、恢复、回放和验证入口。该任务当前使用 MuJoCo owner 配置，后端选择应通过 `task=go2_arm_manip_loco/mujoco` 或统一 CLI 的 `--sim mujoco` 完成，不要单独覆盖 `training.sim_backend` 来切换后端。

## 适用范围

- 任务名: `Go2ArmManipLoco`
- PPO task owner: `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`
- HIM-PPO task owner: `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`
- 场景入口: `src/unilab/assets/robots/go2_arm/scene_flat.xml`
- 默认日志目录: `logs/rsl_rl_ppo/Go2ArmManipLoco/`

## PPO 训练

常规训练优先使用统一 CLI:

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

需要直接调试 Hydra compose 或底层脚本时，使用 RSL-RL 入口:

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

常用覆盖项:

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco \
  algo.max_iterations=300 \
  algo.num_envs=4096 \
  training.no_play=true
```

## 恢复训练

恢复最新 run:

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco algo.load_run=-1 training.no_play=true
```

恢复指定 run:

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco \
  algo.load_run=2026-05-13_16-07-42_mujoco \
  training.no_play=true
```

## 回放

回放最新 checkpoint:

```bash
uv run eval --algo ppo --task go2_arm_manip_loco --sim mujoco --load-run -1
```

回放指定 run:

```bash
uv run eval --algo ppo --task go2_arm_manip_loco --sim mujoco \
  --load-run 2026-05-13_16-07-42_mujoco
```

MuJoCo 默认会导出 `play_video.mp4`。如果只想验证 checkpoint 选择和环境初始化，可以减少回放步数:

```bash
uv run eval --algo ppo --task go2_arm_manip_loco --sim mujoco --load-run -1 training.play_steps=200
```

## HIM-PPO 入口

HIM-PPO 使用单独配置组和脚本入口:

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

回放 HIM-PPO checkpoint:

```bash
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.play_only=true algo.load_run=-1
```

## 检查项

训练前建议先做一次近风险验证:

```bash
uv run pytest tests/envs/locomotion/go2_arm tests/base/backend/test_mujoco_site_jacobian.py
```

如果改过 XML 或 asset，至少确认 MuJoCo 能加载场景:

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2_arm/scene_flat.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

正常输出应包含 `25 24 18 97`。

## 调参提示

- `env.control_config.arm_action_scale` 控制机械臂 residual action 的幅度。
- `env.goal_ee` 控制末端目标采样范围、轨迹时间和碰撞过滤。
- `reward.scales.tracking_lin_vel`、`tracking_ang_vel`、`stand_still` 直接影响四足跟踪和静止行为的权衡。
- `env.domain_rand` 中的质量、摩擦、推力和 PD 随机化会改变训练难度；排查收敛问题时可以先关闭部分随机化做对照。

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Procedural Terrain](08-procedural-terrain.md)
