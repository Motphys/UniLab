# MLX PPO

语言: 简体中文

MLX PPO 面向 macOS / Apple Silicon，训练脚本仍沿用 PPO 的 task owner 结构。

## 默认入口

```bash
uv run train --algo mlx_ppo --task go2_joystick_flat --sim mujoco
```

## 适用场景

- macOS / Apple Silicon
- 需要 MLX 训练栈

## 实现脚本

MLX PPO 顶层 CLI 会路由到 `scripts/train_mlx_ppo.py`。常规训练用
`uv run train --algo mlx_ppo ...`，checkpoint 回放用
`uv run eval --algo mlx_ppo ...`。

## 当前使用建议

- 想要默认、最稳妥的路径时，优先看 torch PPO。
- 想在 Apple Silicon 上直接跑本地训练时，再切到 MLX PPO。
- 精确支持范围以 [后端支持矩阵](../5-reference/1-backend-support-matrix.md) 为准。

## 日志根目录

```text
logs/mlx_rl_train/<task>/
```

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [PPO (torch / RSL-RL)](1-ppo-torch.md)
- Next: [APPO](3-appo.md)
