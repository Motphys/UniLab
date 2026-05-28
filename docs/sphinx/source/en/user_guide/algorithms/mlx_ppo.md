# MLX PPO

MLX PPO uses the PPO task-owner tree but swaps the training runtime to the MLX
implementation. The entry script is `scripts/train_mlx_ppo.py`, the config is
`conf/ppo/config_mlx.yaml`, and the implementation lives under
`src/unilab/algos/mlx/ppo/`.

## Quick Start

```bash
uv run scripts/train_mlx_ppo.py task=go2_joystick_flat/mujoco
uv run scripts/train_mlx_ppo.py task=go2_joystick_flat/motrix training.no_play=true
```

## Notes

- `conf/ppo/config_mlx.yaml` sets `training.device=mlx`.
- The `mlx` dependency is enabled by the `sys_platform == 'darwin'` marker in
  `pyproject.toml`.
- MLX compose coverage is tracked separately in the generated support matrix:
  {doc}`/zh_CN/user_guide/E-reference/01-backend-support-matrix`.

Use torch PPO first when you need the default training path; use MLX PPO when
you are intentionally exercising the MLX runtime.
