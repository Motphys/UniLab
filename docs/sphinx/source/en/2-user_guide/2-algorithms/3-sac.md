# SAC

SAC is selected through the shared off-policy entrypoint
`scripts/train_offpolicy.py`, which TD3 and FlashSAC share as well. The main
config is `conf/offpolicy/config.yaml`, and the SAC algorithm defaults live in
`conf/offpolicy/algo/sac.yaml`. The current log name is `fast_sac`.

## Runtime Model

The off-policy runner decouples CPU simulation from GPU learning through shared
memory: a collector subprocess fills a CPU-resident replay buffer while the
learner trains on the GPU.

SAC is also the currently validated replay-buffer multi-GPU algorithm. Enable it
with `training.num_gpus > 1`; the host side packs and distributes batches in
parallel, while the GPU learners average gradients. See
{doc}`../1-training/4-multi_gpu` for the full command and constraints.

## Quick Start

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
```

Two-GPU MuJoCo example:

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.max_iterations=500`
- `training.use_amp=true` in the shared off-policy config
- Multi-GPU SAC uses `training.num_gpus=<N>`; this validation round requires
  `algo.obs_normalization=false` and does not support `algo.use_symmetry=true`.

The current runner path in `scripts/train_offpolicy.py` requires synchronized
collection; `training.no_sync_collection=true` is rejected by the script.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
