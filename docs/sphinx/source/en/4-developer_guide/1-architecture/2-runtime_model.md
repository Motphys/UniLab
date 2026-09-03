# Runtime Model

The detailed runtime contract is in
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` and
{doc}`/zh_CN/4-developer_guide/0-index`. This page keeps the English
summary close to the code paths.

## Two Runtime Shapes

### Synchronous PPO Paths

`src/unilab/scripts/train_rsl_rl.py` composes Hydra config,
calls registry bootstrap, constructs the env through `registry.make(...)`, and runs
the learner in the same process. The RSL-RL path adapts `NpEnv` through
`src/unilab/training/rsl_rl.py`.

### Async APPO And Off-Policy Paths

APPO and off-policy runners use a CPU-sim-to-learner split:

```text
CPU physics env loop -> shared IPC buffer -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO uses `APPORunner`, `RolloutRingBuffer`, and `SharedWeightSync`.
- SAC, TD3, and FlashSAC use one off-policy execution path: `ReplayBuffer`
  provides bounded host ingress, the complete ring lives on one CUDA/MPS
  learner device, and `SharedWeightSync` publishes actor weights.
- `AsyncRunner` in `uni_rl.ipc.async_runner` (uni_rl repo) owns collector process
  startup, stop signaling, and shared-resource cleanup.

## Boundary Rules

- The env remains numpy/vectorized and returns `NpEnvState`.
- GPU tensors and optimizer state belong to learner code, not env code.
- Collector/learner protocols must reuse the existing IPC primitives instead of
  creating ad-hoc parallel protocols in scripts.

## Evidence In Repo

- PPO entrypoint: `src/unilab/scripts/train_rsl_rl.py`
- APPO runner: `uni_rl.appo.runner` (uni_rl repo)
- Off-policy runner: `uni_rl.offpolicy.double_buffer_runner` (uni_rl repo)
- IPC primitives: `uni_rl.ipc.async_runner` (uni_rl repo),
  `uni_rl.ipc.rollout_ring_buffer` (uni_rl repo), `uni_rl.ipc.replay_buffer` (uni_rl repo),
  `uni_rl.ipc.weight_sync` (uni_rl repo)
