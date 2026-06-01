# NaN 可视化工具

PPO 在 `conf/ppo/config.yaml` 的 `training.nan_guard` 下设有一个 NaN guard。启用后，`scripts/train_rsl_rl.py` 会安装 `NanGuard`，检查观测字典和奖励，并在检测到 NaN/Inf 值时写入一个 `.npz` dump 以及模型元数据。

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.nan_guard.enabled=true \
  training.nan_guard.output_dir=/tmp/unilab/nan_dumps
```

viewer 的实现是 `src/unilab/tools/viz_nan.py`，注册为 `unilab-viz-nan` 控制台入口。它会回放一个 dump 路径，并让你选择环境索引。dump 格式和往返加载由 `tests/test_nan_guard.py` 覆盖。
