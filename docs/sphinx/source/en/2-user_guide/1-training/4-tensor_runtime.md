# Tensor runtime

SAC, FlashSAC, and WarpSAC resolve their spawn-facing tensor-runtime settings
before probing the environment or constructing a learner. The owner YAML values
are collected into one bounded `TensorRuntimeSettings` object and passed to the
off-policy runner. Invalid values, unsupported bounds, and impossible memory
combinations therefore fail before collector or learner processes are launched.

This contract covers the scoped single-GPU off-policy runtime. It is not an
autotuner and does not add multi-GPU scaling.

## Owner defaults and bounds

| Setting | Default | Valid range or constraint |
| --- | ---: | --- |
| `training.inference_slot_capacity` | `1` | Positive integer, at most `16` |
| `training.collector_metrics_interval` | `1` | Positive integer, at most `10000` |
| `training.replay_ingress_depth` | `2` | Positive integer, at most `16` |
| `training.replay_ingress_slot_rows` | `null` | `1` through `algo.num_envs`; `null` means `algo.num_envs` |
| Learner rows per synchronization | `algo.batch_size * algo.updates_per_step` | Constrained by the CUDA memory budget |

Values must be exact positive integers. Booleans, strings, floating-point
numbers, zero, and negative values fail closed. The G1 Motion Tracking / MJWarp
owner intentionally overrides only `collector_metrics_interval` to `100`; all
other tensor-runtime defaults remain as listed above.

## Role scheduling

The off-policy owners also expose advisory CPU-scheduling requests for the
learner, device-replay buffer worker, and collector:

```yaml
training:
  learner_scheduling: {nice: null, cpu_ids: null}
  buffer_scheduling: {nice: null, cpu_ids: null}
  collector_scheduling: {nice: null, cpu_ids: null}
```

`nice` is an additive, non-negative request (`0` keeps normal priority; only a
privileged process can lower it later). `cpu_ids` is an optional Linux affinity
request. Both fields are advisory: a container or host policy may reject the
request, in which case training continues and the runtime manifest records the
failed application. Platform support differs (notably for per-thread buffer
affinity), so benchmark on the target host rather than assuming portability.

Use `training.dp_collector_cpu_ids` for the existing multi-rank collector
partition because that contract also sizes backend worker pools. The
`*_scheduling.cpu_ids` fields are for explicit role-priority experiments.

## Replay ingress tradeoffs

The default `replay_ingress_slot_rows: null` resolves to `algo.num_envs` and
publishes one ingress chunk per collector vector. This keeps the publication
and synchronization pattern predictable and is the recommended production
default.

Setting the field to a smaller value reduces resident ingress memory by splitting
one collector vector into contiguous chunks. It is intended only for explicit
memory-pressure tuning. More chunks mean more publications, more opportunities
for producer backpressure, and—on CUDA—one current-stream barrier per published
chunk. Always benchmark a smaller value on the target workload before adopting
it.

Ingress diagnostics count publication chunks, not complete logical collector
vectors. A shutdown may retain a valid row prefix of a partially published
vector; it never publishes an invalid transition row. See {doc}`3-logging` for
the replay-ingress counters and shutdown schema.

## Runtime evidence

The runtime manifest records the effective evidence needed to audit a run:

- `runtime_limits` contains configured, default, effective, and maximum values
  for the resolved tensor-runtime knobs.
- `inference_memory_budget` records the bounded CUDA inference-ring budget.
- `tensor_memory_budget` records the combined CUDA inference, replay storage,
  replay ingress, learner batch, and workspace budget calculation.

The manifest is written before spawn when a budget decision is made. If an
unsafe combination is rejected, the error identifies the offending setting and
the process is not launched.

For the single-GPU G1 FlashSAC/MJWarp production workflow, benchmark
reproduction, artifacts, and troubleshooting, see
{doc}`7-tensor_runtime_production`.
