# SAC

SAC runs through `scripts/train_sac.py`; TD3 and FlashSAC have their own
entrypoints and per-algorithm config trees. The main config is
`conf/sac/config.yaml`, with the SAC algorithm defaults inlined there. The
current log name is `fast_sac`.

## Runtime Model

The off-policy runner decouples CPU simulation from accelerator learning through
bounded shared memory. A collector subprocess publishes packed transitions
through two ingress slots, while the complete replay ring is authoritative on
one CUDA or Apple MPS learner device. Host replay allocation therefore does not
grow with replay capacity; `ptr` and `size` advance only after a slot's device
copy completes. CUDA commits on a side stream, while MPS device work is
submitted only by the learner thread to avoid background Metal submission.
CPU and XPU training are unsupported; there is no alternate replay pipeline.

## Quick Start

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
```

## Key Fields

For the off-policy playback path (`scripts/train_sac.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` is the learner batch per update.
- `algo.max_iterations=500`
- `training.use_amp=true` in `conf/sac/config.yaml`

The off-policy device replay path uses synchronized, learner-owned inference:
collectors exchange observations and actions through shared memory and do not own an actor.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
