# Runtime Model

The detailed runtime contract is in
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` and
{doc}`/zh_CN/2-developer_guide/1-development-standard`. This page keeps the English
summary close to the code paths.

## Two Runtime Shapes

### Synchronous PPO Paths

`scripts/train_rsl_rl.py` and `scripts/train_mlx_ppo.py` compose Hydra config,
call registry bootstrap, construct the env through `registry.make(...)`, and run
the learner in the same process. The RSL-RL path adapts `NpEnv` through
`src/unilab/training/rsl_rl.py`; the MLX path uses
`src/unilab/algos/mlx/ppo/runner.py` and `src/unilab/algos/mlx/ppo/ppo.py`.

### Async APPO And Off-Policy Paths

APPO and off-policy runners use a CPU-sim-to-learner split:

```text
CPU physics env loop -> shared IPC buffer -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO uses `APPORunner`, `RolloutRingBuffer`, and `SharedWeightSync`.
- SAC, TD3, and FlashSAC use off-policy runners with `ReplayBuffer` and
  `SharedWeightSync`.
- `AsyncRunner` in `src/unilab/ipc/async_runner.py` owns collector process
  startup, stop signaling, and shared-resource cleanup.

## Boundary Rules

- The env remains numpy/vectorized and returns `NpEnvState`.
- GPU tensors and optimizer state belong to learner code, not env code.
- Collector/learner protocols must reuse the existing IPC primitives instead of
  creating ad-hoc parallel protocols in scripts.

## Evidence In Repo

- PPO entrypoints: `scripts/train_rsl_rl.py`, `scripts/train_mlx_ppo.py`
- APPO runner: `src/unilab/algos/torch/appo/runner.py`
- Off-policy runner: `src/unilab/algos/torch/offpolicy/runner.py`
- IPC primitives: `src/unilab/ipc/async_runner.py`,
  `src/unilab/ipc/rollout_ring_buffer.py`, `src/unilab/ipc/replay_buffer.py`,
  `src/unilab/ipc/weight_sync.py`
