# PPO

PPO is the default synchronous on-policy training path. It uses
`src/unilab/scripts/train_rsl_rl.py`, composes from `src/unilab/conf/ppo/config.yaml`, and runs the
RSL-RL adapter code in `uni_rl.algos.rsl_rl_ppo` (unilab-rl repo) and
`src/unilab/training/rsl_rl.py`.

## Quick Start

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## Common Overrides

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

Use `uv run eval` for checkpoint playback:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Logs are grouped by `algo.algo_log_name`; the default in `src/unilab/conf/ppo/config.yaml`
is `rsl_rl_ppo`.

## Single-node multi-GPU training

`training.devices` enables RSL-RL's synchronous data-parallel PPO path:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  'training.devices=[0,1]' \
  training.no_play=true

uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  'training.devices=[0,1]' \
  training.no_play=true
```

`null` or `[]` keeps automatic single-device selection, `[d]` selects one
CUDA device, and two or more entries launch one local process per listed
device. Do not set `training.device` and `training.devices` together. The
configured order is preserved, including when the parent already has
`CUDA_VISIBLE_DEVICES` set.

For the IsaacGym, IsaacSim, and Genesis owners, the same topology is also
applied to the simulator environment. Torchrun workers receive the local
index inside their remapped `CUDA_VISIBLE_DEVICES` list (for example, host
device 5 is sent as `device_id=1` when the worker sees `[4,5]`); off-policy
workers keep the parent process's visible index namespace. Genesis selects
its process-wide session before `gs.init`.

`algo.num_envs` is a **per-rank** count, not a global budget. For `W` ranks,
`N` configured envs, and rollout length `T`:

```text
local samples / iteration  = N * T
global samples / iteration = W * N * T
```

Each rank owns an independent env, policy copy, and rollout storage. Rollouts,
GAE, advantage normalization, and mini-batch shuffling stay rank-local.
RSL-RL broadcasts rank 0's model state at startup and averages gradients after
every PPO mini-batch backward pass. Rank `i` uses `algo.seed + i`.
`training.num_timesteps` is interpreted as a global sample budget and therefore
uses `W * N * T` when deriving the iteration count.

Only rank 0 writes TensorBoard/W&B data, checkpoints, `run_config.json`, and
`run_summary.json`, then optionally enters playback after all ranks have closed
their process group. `run_summary.json` records `world_size`, per-rank/global
env counts, samples per iteration, and aggregate training throughput. RSL-RL's
`Perf/total_fps` is also an aggregate global samples/s metric; reward and episode
statistics remain rank-0-local.

RSL-RL does not synchronize observation-normalizer buffers or environment
curriculum state after startup. Consequently, tasks with empirical
normalization (including the current Go2 flat owner) keep rank-local statistics,
and the checkpoint contains rank 0's copy. This is the upstream RSL-RL
distributed semantic, not global-rollout PPO.

The integrated launcher is single-node only. The repository has two-GPU MuJoCo
smoke coverage for `go2_joystick_flat` and `g1_motion_tracking`; this does not
claim multi-node or Motrix multi-GPU support. On the currently validated RTX
6000D host, the launcher inherits the production NCCL compatibility defaults
`NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1`; explicit environment values win.
