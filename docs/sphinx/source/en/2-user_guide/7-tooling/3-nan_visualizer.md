# NaN Visualizer

PPO has a NaN guard under `training.nan_guard` in `conf/ppo/config.yaml`. When
enabled, `scripts/train_rsl_rl.py` installs `NanGuard`, checks observation dicts
and rewards, and writes a `.npz` dump plus model metadata when it detects
NaN/Inf values.

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.nan_guard.enabled=true \
  training.nan_guard.output_dir=/tmp/unilab/nan_dumps
```

The viewer implementation is `src/unilab/tools/viz_nan.py`, registered as the
`unilab-viz-nan` console entry. It replays a dump path and lets you select the
environment index. Dump format and round-trip loading are covered by
`tests/test_nan_guard.py`.
