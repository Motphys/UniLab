# PPO (torch / RSL-RL)

语言: 简体中文

RSL-RL PPO 是 UniLab 默认、最直接的训练入口。

## 默认入口

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

## 适用场景

- 默认首选训练路径
- 想先用最稳定、最直接的单机训练入口

## 常见覆盖

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

## 关联脚本

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
```

## 日志根目录

```text
logs/rsl_rl_ppo/<task>/
```

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [算法说明](../04-algorithms.md)
- Next: [MLX PPO](02-mlx-ppo.md)
