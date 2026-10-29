---
orphan: true
---

# ADR-0020 FlashSAC DP Graph Ownership And Wire Precision

- Status: Proposed
- Date: 2026-10-22
- Owners: UniLab RL runtime maintainers
- Supersedes: None
- Superseded by: None

## Context

FlashSAC disables whole-cycle capture when gradient synchronization is attached,
although the distributed runner already provides graph lifecycle hooks. Multi-GPU
profiling also shows that large gradient reductions dominate learner time on
hosts without GPU peer access. Downstream FlashMimic should consume these runtime
capabilities rather than patching algorithm or communication implementations.

## Decision

Keep the existing runner and process-group owner. Enable FlashSAC DP whole-cycle
capture in the owner YAML by default, with an explicit
`algo.algo_params.dp_whole_cycle_cuda_graph=false` opt-out, and expose a separate
runtime wire-precision option. Preserve FP32 communication as the default. This decision
does not change physics backend contracts or establish BF16 convergence parity.

## Stable Contracts

- `algo.algo_params.dp_whole_cycle_cuda_graph=true` requires synchronizer hooks;
  changing hooks, gradient callbacks, or checkpoint state invalidates capture.
- DP capture errors propagate. A rank must not independently fall back to eager
  collectives. Graphs are released before process-group destruction.
- Compiled CUDA execution without GradScaler includes loss validity in gradient
  reduction, so all ranks skip the same invalid optimizer and critic target step.
- `training.dp_gradient_wire_dtype` accepts `float32` (default) or `bfloat16`.
  BF16 affects only FP32 groups with at least 1024 elements. Restore FP32 gradients
  before division; preserve parameter/optimizer dtype and gradient storage.
- Every rank uses the same layout and precision. The option is independent of AMP.
- MPS pipes/logs are selected per rank before CUDA initialization. Explicit host
  daemon directories do not require access to another user's default runtime root.

## Alternatives Considered

- Keep graph execution disabled: retains avoidable learner launch overhead.
- Enable BF16 by default: rejected because gradient reduction is lossy and
  long-run convergence equivalence has not been established.
- Fuse actor and temperature reductions: excluded after no stable measured
  throughput improvement; it would also change the small-group precision boundary.
- Put optimized runtime code in FlashMimic: rejected because UniLab owns DP.

## Consequences

Optional CUDA/NCCL execution requires real multi-rank validation in addition to
CPU regression tests. Historical throughput evidence must identify its exact
source pins and configuration; it is not a promise for newer backend defaults.
Capturing a raw CUDA cycle does not imply one fully compiled learner FX graph.

## Evidence In Repo

- `src/unilab/rl/algos/flash_sac/learner.py`
- `src/unilab/rl/runtime/distributed/sync.py`
- `src/unilab/rl/runtime/offpolicy/double_buffer_runner.py`
- `src/unilab/scripts/train_offpolicy.py`
- `tests/rl/algos/test_flash_sac_dp_graph.py`
- `tests/rl/runtime/test_dp_wire_precision.py`
- `tests/algos/test_offpolicy_dp_sync.py`
- `tests/training/test_cuda_mps_cli.py`

## Related Documents

- {doc}`ADR-0016-rl-reintegration-into-unilab-rl`
- {doc}`ADR-0014-cuda-mps-cli-lifecycle-owner`
