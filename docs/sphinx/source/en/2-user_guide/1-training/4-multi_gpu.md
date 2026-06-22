# Multi-GPU

The currently validated multi-GPU training path is SAC in replay-buffer mode.
Use the unified CLI as usual, and enable multiple GPUs with the shared
off-policy field `training.num_gpus`.

The multi-GPU runner keeps algorithm code separate from IPC: a collector fills
the CPU replay buffer on the host, the runner packs batches for each learner
rank on demand, pinned-memory pipelines distribute those batches to multiple
GPUs in parallel. The collector remains a single CPU collector; multi-GPU mode
does not implicitly increase `algo.num_envs`.

## Learner Synchronization

Multi-GPU SAC defaults to `training.multi_gpu_sync_mode=local_sgd`. Each learner
rank applies local SAC updates on its own GPU without synchronizing gradients
after every critic / actor update. At the runner-controlled iteration boundary,
the ranks average actor, critic, target critic, and entropy-coefficient
parameters according to `training.multi_gpu_sync_interval`; at that sync
boundary rank 0 publishes the averaged actor weights to the CPU collector. The default
`training.multi_gpu_sync_interval=1` synchronizes after every learner iteration.
Increasing it further reduces communication on 4-GPU and 8-GPU runs, at the
cost of larger inter-rank parameter drift. In `local_sgd` mode, optimizer state
is intentionally rank-local and is not averaged at the synchronization boundary;
this avoids extending communication to AdamW momentum state.

For strict per-update gradient averaging, set
`training.multi_gpu_sync_mode=sync_sgd`. That mode is closer to single-GPU
global-batch semantics, but it performs many more collectives and is usually a
poor performance fit on systems without fast GPU interconnects.

## Batch Semantics

For off-policy SAC, `algo.batch_size` is the batch size **per learner rank per
update**, not the global batch across all GPUs. With `training.num_gpus=N`, the
effective global update batch is `algo.batch_size * N`. Each rank independently
samples its own batch from the shared replay buffer, then gradients are averaged
with distributed training.

As a result, a two-GPU run with the same `algo.batch_size` uses twice as many
effective samples per update as a single-GPU run. In the default `local_sgd`
mode those local updates are parameter-averaged at the iteration boundary; in
`sync_sgd` mode gradients are averaged after each update. To keep the global
update batch unchanged, scale `algo.batch_size` down manually; for example,
single-GPU `algo.batch_size=8192` corresponds to two-GPU
`algo.batch_size=4096`. Logger field `Batch/Rank` is the per-rank batch, while
`Batch/Update` is the global batch.

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
  training.multi_gpu_sync_mode=local_sgd \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

For non-adjacent physical GPUs, map them into the visible set with
`CUDA_VISIBLE_DEVICES`. For example, to use physical cards 0 and 7:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

For a short smoke run, reduce iterations and env count, and skip post-training
playback:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd \
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
logger, and visible GPUs consistent; also decide whether you are comparing the
same per-rank batch or the same global batch. Then compare steady-state
`train_fps`, learner step time, and end-to-end iteration time.

## Common Errors

- `Only SAC supports training.num_gpus > 1`: only SAC is validated right now.
- `SAC multi-GPU training requires a CUDA device`: CUDA is unavailable, or
  `training.device` was set to CPU.
- `requires algo.obs_normalization=false`: add `algo.obs_normalization=false`.
- `set training.num_gpus=1 or algo.use_symmetry=false`: multi-GPU SAC does not
  support symmetry augmentation yet; add `algo.use_symmetry=false`.

When changing multi-GPU behavior, validate near the off-policy runner and IPC
boundary rather than only checking a top-level command.
