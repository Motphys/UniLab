# Multi-GPU

The current multi-GPU knob lives in the shared off-policy training config as
`training.num_gpus`. The field is consumed by the off-policy and FlashSAC paths;
PPO, MLX PPO, and APPO do not expose the same multi-GPU contract.

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco \
  training.num_gpus=2 \
  training.no_play=true
```

Keep task and backend selection in the owner YAML path:

```bash
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco \
  training.num_gpus=2
```

When changing multi-GPU behavior, validate near the off-policy runner and IPC
boundary rather than only checking a top-level command.
