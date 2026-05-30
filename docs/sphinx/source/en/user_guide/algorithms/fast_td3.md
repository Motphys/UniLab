# FastTD3

FastTD3 shares the off-policy training script with SAC and FlashSAC. Select it
with `algo=td3` and an owner path under `conf/offpolicy/task/td3/`.

## Quick Start

```bash
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo td3`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`../getting_started/evaluation_and_playback`.

- Defaults live in `conf/offpolicy/algo/td3.yaml`.
- `algo.algo_log_name=fast_td3`.
- `algo.max_iterations=5000`.
- `algo.policy_frequency=2`.

Use the owner path to select task and backend; do not reuse a SAC owner with
`algo=td3`.

```bash
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco \
  algo.num_envs=2048 \
  training.no_play=true
```
