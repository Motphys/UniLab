# SAC

SAC is selected through the shared off-policy entrypoint
`scripts/train_offpolicy.py`, which TD3 and FlashSAC share as well. The main
config is `conf/offpolicy/config.yaml`, and the SAC algorithm defaults live in
`conf/offpolicy/algo/sac.yaml`. The current log name is `fast_sac`.

## Runtime Model

The off-policy runner decouples CPU simulation from GPU learning through shared
memory: a collector subprocess fills a CPU-resident replay buffer while the
learner trains on the GPU.

Single-GPU CUDA and Apple MPS runs may select
`training.replay_pipeline=gpu_resident`. In this single-device path, the full
replay ring is authoritative on the learner device. The collector publishes
packed transitions through two bounded shared-memory ingress slots, so host
replay allocation does not grow with replay capacity; `ptr` and `size` advance
only after a slot's device copy completes. The default remains
`cpu_pinned_double_buffer`. CUDA commits on a side stream, while MPS device work
is submitted only by the learner thread to avoid background Metal submission.
The multi-GPU `gpu_resident` migration is separate and still uses its existing
CPU-authoritative mirror path.

The default FastSAC learner is also the currently validated replay-buffer
multi-GPU SAC implementation. Enable it with `training.num_gpus > 1`; the host
side packs and distributes batches in parallel, while the GPU learners default
to delayed parameter averaging via `training.multi_gpu_sync_mode=local_sgd`.
Custom SAC runtimes must explicitly declare the distributed learner contract
before they can use this path. See
{doc}`../1-training/4-multi_gpu` for the full command, strict-sync fallback, and
constraints.

## Quick Start

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.replay_pipeline=gpu_resident
```

Two-GPU MuJoCo example:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.use_symmetry=false
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` is the per learner rank batch per update; in multi-GPU
  runs, the global update batch is `algo.batch_size * training.num_gpus`.
- `algo.max_iterations=500`
- `training.use_amp=true` in the shared off-policy config
- Multi-GPU SAC uses `training.num_gpus=<N>`; `algo.use_symmetry=true` is not
  supported yet.
- Multi-GPU SAC defaults to `training.multi_gpu_sync_mode=local_sgd` and
  `training.multi_gpu_sync_interval=1`.

Single-GPU SAC and TD3 can run the off-policy double-buffer path with
`training.no_sync_collection=true`. Multi-GPU SAC and FlashSAC-B still require
synchronized collection.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
