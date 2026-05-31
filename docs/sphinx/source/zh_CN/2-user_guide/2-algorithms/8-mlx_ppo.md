# MLX PPO

MLX PPO 使用 PPO 的 task-owner 树，但将训练运行时替换为 MLX 实现。入口脚本是
`scripts/train_mlx_ppo.py`，配置是 `conf/ppo/config_mlx.yaml`，实现位于
`src/unilab/algos/mlx/ppo/` 下。

## 快速开始

```bash
uv run train --algo mlx_ppo --task go2_joystick_flat --sim mujoco
uv run train --algo mlx_ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## 说明

- `conf/ppo/config_mlx.yaml` 设置 `training.device=mlx`。
- `mlx` 依赖由 `pyproject.toml` 中的 `sys_platform == 'darwin'` marker 启用。
- MLX 的 compose 覆盖情况在生成的支持矩阵中单独跟踪：
  {doc}`/zh_CN/5-reference/5-support_matrix`。

当你需要默认训练路径时优先使用 torch PPO；当你有意运行 MLX 运行时时再使用
MLX PPO。
