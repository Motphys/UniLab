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

## 实现脚本

PPO 顶层 CLI 会路由到 `scripts/train_rsl_rl.py`。常规训练用上面的
`uv run train --algo ppo ...`，checkpoint 回放用 `uv run eval --algo ppo ...`。

## 日志根目录

```text
logs/rsl_rl_ppo/<task>/
```

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [算法说明](../4-algorithms.md)
- Next: [MLX PPO](2-mlx-ppo.md)
