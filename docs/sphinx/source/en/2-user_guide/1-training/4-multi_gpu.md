# Multi-GPU

The currently validated multi-GPU training path is SAC in replay-buffer mode.
Use the unified CLI as usual, and enable multiple GPUs with the shared
off-policy field `training.num_gpus`.

The multi-GPU runner keeps algorithm code separate from IPC: a collector fills
the CPU replay buffer on the host, the runner packs batches for each learner
rank on demand, pinned-memory pipelines distribute those batches to multiple
GPUs in parallel, and the GPU learners average gradients with distributed
training.

## Preconditions

- SAC only: `training.num_gpus > 1` rejects TD3, FlashSAC, PPO, MLX PPO, and APPO.
- CUDA is required; select physical cards with `CUDA_VISIBLE_DEVICES`.
- This validation round requires `algo.obs_normalization=false`.
- SAC symmetry augmentation is not supported in multi-GPU mode. If the task
  owner enables it by default, set `algo.use_symmetry=false`.
- Collection must stay synchronized; do not set `training.no_sync_collection=true`.

## Basic Command

Two adjacent visible GPUs:

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

For non-adjacent physical GPUs, map them into the visible set with
`CUDA_VISIBLE_DEVICES`. For example, to use physical cards 0 and 7:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

For a short smoke run, reduce iterations and env count, and skip post-training
playback:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false \
  algo.max_iterations=10 \
  algo.num_envs=512 \
  training.no_play=true
```

Logs still use SAC's default directory: `logs/fast_sac/<TaskName>/`.

## Performance Checks

Multi-GPU mainly targets learner update bottlenecks. For small env counts,
batches, or short runs, distributed startup, batch packing, and gradient
synchronization can cost more than they save. When comparing single-GPU and
multi-GPU runs, keep the task, env count, iteration count, playback settings,
logger, and visible GPUs consistent, then compare steady-state `train_fps`,
learner step time, and end-to-end iteration time.

## Common Errors

- `Only SAC supports training.num_gpus > 1`: only SAC is validated right now.
- `SAC multi-GPU training requires a CUDA device`: CUDA is unavailable, or
  `training.device` was set to CPU.
- `requires algo.obs_normalization=false`: add `algo.obs_normalization=false`.
- `set training.num_gpus=1 or algo.use_symmetry=false`: multi-GPU SAC does not
  support symmetry augmentation yet; add `algo.use_symmetry=false`.

When changing multi-GPU behavior, validate near the off-policy runner and IPC
boundary rather than only checking a top-level command.
