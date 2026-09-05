# SAC

SAC runs through `src/unilab/scripts/train_sac.py`; TD3 and FlashSAC have their own
entrypoints and per-algorithm config trees. The main config is
`src/unilab/conf/sac/config.yaml`, with the SAC algorithm defaults inlined there. The
current log name is `fast_sac`.

## Runtime Model

The off-policy runner decouples simulation collection from accelerator learning through
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

For the off-policy playback path (`src/unilab/scripts/train_sac.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` is the learner batch per update.
- `algo.max_iterations=500`
- `training.use_amp=true` in `src/unilab/conf/sac/config.yaml`

The off-policy device replay path uses synchronized, learner-owned inference:
collectors exchange observations and actions through shared memory and do not own an actor.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```

## Single-node multi-GPU device placement

`training.devices` assigns rank i's learner to `cuda:devices[i]`; each rank owns one
collector. For mjwarp, the rank process and its collector process explicitly bind Warp's
default/current device to that same learner device before probe or production environment
materialization. The collector therefore does not fall back to Warp's fresh-process default
of `cuda:0`. The local binding is recorded as `collector_backend_device` in the runtime
manifest.

IsaacGym, IsaacSim, and Genesis receive the rank-selected simulator device
through the environment override as well. Off-policy collectors use the
parent's visible CUDA indices; Genesis binds its process-wide session before
initialization.

MuJoCo has a committed multi-GPU scaling benchmark. The mjwarp per-rank placement contract is
covered by `tests/base/backend/test_process_device.py` and the off-policy runner/worker unit
tests; the repository does not currently contain an mjwarp multi-GPU throughput or convergence
benchmark.
