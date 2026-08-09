# SAC

SAC is selected through the shared off-policy entrypoint
`scripts/train_offpolicy.py`, which TD3 and FlashSAC share as well. The main
config is `conf/offpolicy/config.yaml`, and the SAC algorithm defaults live in
`conf/offpolicy/algo/sac.yaml`. The current log name is `fast_sac`.

## Runtime Model

The off-policy runner decouples CPU simulation from GPU learning through shared
memory: a collector subprocess fills a CPU-resident replay buffer while the
learner trains on the GPU.

CUDA and Apple MPS runs may select
`training.replay_pipeline=gpu_resident`. In this single-device path, the full
replay ring is authoritative on the learner device. The collector publishes
packed transitions through two bounded shared-memory ingress slots, so host
replay allocation does not grow with replay capacity; `ptr` and `size` advance
only after a slot's device copy completes. The default remains
`cpu_pinned_double_buffer`. CUDA commits on a side stream, while MPS device work
is submitted only by the learner thread to avoid background Metal submission.
Each SAC training process uses one learner device.

## Quick Start

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.replay_pipeline=gpu_resident
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` is the learner batch per update.
- `algo.max_iterations=500`
- `training.use_amp=true` in the shared off-policy config

SAC and TD3 can run the off-policy double-buffer path with
`training.no_sync_collection=true`. FlashSAC-B requires synchronized collection.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
