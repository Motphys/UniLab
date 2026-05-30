# FastSAC

FastSAC is selected through the shared off-policy entrypoint
`scripts/train_offpolicy.py`. The main config is `conf/offpolicy/config.yaml`,
and the SAC algorithm defaults live in `conf/offpolicy/algo/sac.yaml`.

## Quick Start

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_rough/motrix training.no_play=true
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`../getting_started/evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.max_iterations=500`
- `training.use_amp=true` in the shared off-policy config

The current runner path in `scripts/train_offpolicy.py` requires synchronized
collection; `training.no_sync_collection=true` is rejected by the script.

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
