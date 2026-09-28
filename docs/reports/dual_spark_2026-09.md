# Dual-Spark Throughput Scaling Report (2026-09)

## Environment

| Item | Value |
| --- | --- |
| Hosts | 2 × NVIDIA Spark (GB10, one CUDA GPU per host, 128 GB unified memory) |
| Link | QSFP direct, 200 Gb/s negotiated, ~95 Gbit/s measured TCP |
| Transport | NCCL over TCP (`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1`) |
| PPO launch | `torchrun --nnodes=2 --nproc_per_node=1` |
| Off-policy launch | uni_rl external DP topology (TCPStore rendezvous) |
| Backend | MuJoCo |
| Software | UniLab `feat/dual-spark` (based on 838cc03e); unilab_rl `feat/dual-spark` (based on 1.3.2); torch 2.9.0+cu130 |

## Methodology

Each Spark runs at the **full single-machine training scale** (same
`num_envs` and `batch_size` as the single-node baseline). Gradients are
all-reduced after every update, keeping both hosts bitwise-identical.
System throughput:

- **PPO**: global env steps/s (`Perf/total_fps`, aggregated across world_size)
- **off-policy**: global learner samples/s (`global batch × updates/iter ÷ iter_ms`)

500 iterations per run (50-iteration warmup excluded). Same seed and
hyperparameters on both sides.

## PPO throughput

| Task | Per-node envs | Single (steps/s) | Dual (steps/s) | **Speedup** |
| --- | --- | --- | --- | --- |
| g1_walk_flat | 2048 | 43,558 | 81,620 | **1.87×** |
| g1_flip_tracking | 1024 | 23,244 | 41,354 | **1.78×** |
| go2_joystick_flat | 2048 | 68,659 | 132,516 | **1.93×** |
| allegro_inhand | 16384 | 46,956 | 92,671 | **1.97×** |

Speedup approaches 2× as the environment count grows: rollout collection
dominates PPO on Spark and shards cleanly.

## Off-policy throughput

Per-node batch = 8192 (same as single-node baseline).

| Workload | Per-node scale | Single iter | Dual iter | Single (samples/s) | Dual (samples/s) | **Speedup** |
| --- | --- | --- | --- | --- | --- | --- |
| SAC g1_walk_flat | 2048 env / b8192 | 91.0ms | 127.9ms | 540,063 | 768,403 | **1.42×** |
| SAC g1_motion_tracking | 2048 env / b8192 | 64.5ms | 92.0ms | 423,371 | 593,463 | **1.40×** |
| **FlashSAC g1_walk_flat** | 4096 env / b8192 | 901.4ms | 945.4ms | 48,471 | 92,430 | **1.91×** |
| **FlashSAC g1_motion_tracking** | 2048 env / b8192 | 471.2ms | 494.7ms | 46,360 | 88,309 | **1.90×** |

FlashSAC's dual-node iteration time is only ~5% slower while consuming 2×
samples per iteration → near-perfect 2× throughput. SAC's small-MLP learner
is kernel-launch-bound; the ~38ms of DP overhead (gradient sync + wrapper)
is 30% of a 91ms iteration, limiting throughput to 1.42×.

## Why the difference: bottleneck analysis

### Single-node bottleneck (SAC g1_walk @ b8192, TB segment timing)

| Segment | Single | Dual | Notes |
| --- | --- | --- | --- |
| **total iteration** | **91.0ms** | **127.9ms** | dual 40% slower |
| L train (updates incl. sync) | 89.3ms | 126.8ms | 98%+ of iteration |
| L — gradient all-reduce | — | ~12.0ms | 18 × ~0.55ms |
| C active collection | 33.0ms | 18.5ms | sharding 0.56× |
| C idle wait for learner | 61.4ms | 114.8ms | collection has ~50% slack |

### Dual-node cost decomposition (+36.9ms on 91.0ms)

| Item | Value | Source |
| --- | --- | --- |
| single b8192 iter | 91.9ms | smoke |
| single b4096 iter | 80.2ms | halving the batch saves only 11.7ms (under-saturated) |
| dual b8192/node iter | 127.9ms | measured |
| gradient sync | +12.0ms | `dp_sync_time` (18 × 0.55ms) |
| DP wrapper overhead | +25.9ms | flat-gradient packing + sync breaking async pipeline |
| net (batch unchanged) | 0 + 12.0 + 25.9 = +37.9ms | matches 127.9 − 91.0 |

### Why FlashSAC nearly doubles

FlashSAC's distributional learner computes 6–10× more per update (batch 8192:
~901ms vs SAC's ~91ms). The DP overhead (~12ms sync + ~26ms wrapper) is <5%
of the iteration. Each rank does exactly what a single node does, so
iteration time barely changes and throughput ≈ 2 × 901/945 = **1.91×**.

### GB10 batch saturation curve

| Global batch | Single iter | Notes |
| --- | --- | --- |
| 2048 | 72.4ms | under-saturated: doubling adds only +10ms |
| 4096 | 82.0ms | same |
| 8192 | 91.9ms | knee point |
| 16384 | 145.4ms | saturated: doubling adds +53ms |

## Conclusions

1. **Per-node full-scale throughput scaling is the best use of a second
   Spark**: FlashSAC 1.91×, PPO 1.97× at large env counts.
2. **The boundary is clear**: heavy learners (large batch, distributional
   critic, large networks) → near 2×; light learners (small MLP + small
   batch) → ~1.4×, limited by DP fixed overhead as a fraction of iteration
   time.
3. **Larger networks (transformer policy/critic) will further raise the
   compute/communication ratio**, pushing closer to linear scaling.
