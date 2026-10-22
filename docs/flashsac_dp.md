# FlashSAC data-parallel CUDA graphs and wire precision

## Scope and decision

This change ports the measured FlashMimic multi-GPU optimization to UniLab
1.5.0 (`03b75353`) with UniSim 2.1.0 (`6994cd1`). The intended PR base is `main`.
The requested scope is default DP graph execution and optional BF16 gradient
communication; the existing runner owns process lifecycle and synchronization.
No new runner or backend protocol is introduced. Communication remains FP32;
DP graphs default on, with explicit opt-out. The actor/temperature fusion experiment is excluded because
it did not provide a stable throughput improvement.

The proposed contract is recorded in
[ADR-0020](sphinx/source/adr/ADR-0020-flashsac-dp-graph-and-wire-precision.md).

## Execution contract

FlashSAC captures the update cycle with NCCL reductions by default on supported
compiled CUDA execution. Set `algo.algo_params.dp_whole_cycle_cuda_graph=false`
to disable DP capture. The runner attaches synchronizer hooks
around capture and before replay. Replacing callbacks, hooks, or checkpoint
state invalidates the graph. A DP capture failure raises rather than allowing
one rank to fall back to eager collectives. A loss-validity sentinel is packed
with each optimizer's gradients, including losses whose derivatives are finite.
On the compiled CUDA path without GradScaler, every rank therefore gates that
optimizer and the critic target consistently. CPU, eager CUDA and FP16 scaler
paths retain their existing finite-check behavior.

This is raw CUDA graph capture around compiled objective/backward segments and
optimizer updates. It is not a claim that the entire learner was compiled into
one `torch.compile(fullgraph=True)` graph.

`training.dp_gradient_wire_dtype=float32` preserves the default reduction.
Explicit `bfloat16` packs FP32 groups of at least 1024 elements in BF16, reduces
their SUM, restores gradients to FP32, then divides in FP32. Smaller groups and
other source dtypes retain their original dtype. Parameters and optimizer states
retain their existing dtype. Layout and precision must match across all ranks.
This is a lossy numerical option, independent of AMP; long-run convergence
equivalence has not been demonstrated.

## Evidence and acceptance

Historical experiments used UniLab `c71a7a20`, UniSim `26d17b4e`, FlashMimic
`d3990125`, with env=2048, batch=2048 per rank and six updates per cycle. On the
tested eight-GPU host, BF16 wire increased median throughput from 346875 to
388227 steps/s (+11.9%). Relative gradient L2 error was approximately 0.3%.
These are historical measurements, not benchmarks of the new upstream defaults.
Learner time still grew with rank count; constant learner time was not achieved.

Acceptance covers default precision, BF16 threshold and mean semantics, stable
buffer storage, graph ownership and failure behavior, rank-wide nonfinite-loss
gates, and Linux multi-GPU capture/replay. README commands show explicit GPU
selection and user-owned MPS. CPU mock tests do not establish NCCL graph support.

## Minimal launch and manual MPS

From UniLab, use `CUDA_VISIBLE_DEVICES=0,1 uv run --extra mjwarp train --algo
flashsac --task g1_motion_tracking --sim mjwarp` for two ranks. Omit the visibility
variable for one rank.
On hosts with broken peer transport, prefix `NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0` to disable P2P and enable shared memory.

MPS remains manually managed. Start one service per selected GPU with
`uv run uni-cumps start --gpus 0 --name train-gpu0 --daemon` (repeat for GPU 1
with its own name), then append `training.cuda_process_sharing=mps` to training.
After training clients exit, stop each service with
`uv run uni-cumps stop --name train-gpu0`. The default `null` uses independent
CUDA contexts. Training does not start or stop services. README documents the
complete examples and optional env, batch, iteration, and transport overrides.
Full repository and current-head remote CI gates remain required before merge.
