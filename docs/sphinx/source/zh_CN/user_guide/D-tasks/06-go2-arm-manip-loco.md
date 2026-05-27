# Go2 Arm Manip Loco

语言: 简体中文

## 任务

- PPO：`go2_arm_manip_loco`
- HIM-PPO：同一任务名，单独脚本入口

## 默认命令

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
uv run eval --algo ppo --task go2_arm_manip_loco --sim mujoco --load-run -1
```

## 配置入口

- PPO owner：`conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`
- HIM-PPO owner：`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`
- 场景 XML：`src/unilab/assets/robots/go2_arm/scene_flat.xml`

## 低层入口

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco training.no_play=true
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

## 恢复训练与回放

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco algo.load_run=-1 training.no_play=true
uv run eval --algo ppo --task go2_arm_manip_loco --sim mujoco --load-run 2026-05-13_16-07-42_mujoco
uv run scripts/train_him_ppo.py task=go2_arm_manip_loco/mujoco training.play_only=true algo.load_run=-1
```

## 近风险检查

```bash
uv run pytest tests/envs/locomotion/go2_arm tests/base/backend/test_mujoco_site_jacobian.py
```

如果改过 XML 或 asset，至少确认 MuJoCo 能加载场景：

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2_arm/scene_flat.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

## 调参提示

- `env.control_config.arm_action_scale`：机械臂 residual action 幅度
- `env.goal_ee`：末端目标采样范围和轨迹时间
- `reward.scales.tracking_lin_vel`、`tracking_ang_vel`、`stand_still`：底盘行为权衡
- `env.domain_rand`：质量、摩擦、推力和 PD 随机化会改变训练难度

## 关联入口

- 训练规则：看 [03 训练指南](../03-training.md)
- 任务总索引：看 [D 任务索引](01-task-index.md)

## Navigation

- Index: [Documentation](../../index.md)
- Previous: [Go2 Rough Terrain](05-go2-rough-terrain.md)
