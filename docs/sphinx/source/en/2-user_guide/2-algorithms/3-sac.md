# SAC

SAC runs through `src/unilab/scripts/train_sac.py`; FlashSAC has its own
entrypoint and per-algorithm config tree. The main config is
`src/unilab/conf/sac/config.yaml`, with the SAC algorithm defaults inlined there. The
current log name is `sac`.

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
```

## Key Fields

For the off-policy playback path (`src/unilab/scripts/train_sac.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=sac`
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

`CUDA_VISIBLE_DEVICES` is the sole GPU topology source. Each parent entry maps to one
rank, and within that rank the learner, collector, inference ring, replay ingress, and
backend payload all use the same physical GPU as local `cuda:0`. Off-policy collectors
never consume another rank's device or a parent-global CUDA ordinal.

IsaacGym, IsaacSim, and Genesis receive the rank-local simulator ordinal through the
environment override. Genesis binds its process-wide session before initialization.

MuJoCo has a committed multi-GPU scaling benchmark. The mjwarp per-rank placement contract is
covered by `tests/base/backend/test_process_device.py` and the off-policy runner/worker unit
tests; the repository does not currently contain an mjwarp multi-GPU throughput or convergence
benchmark.

## Configuration reference

Defaults come from the typed configs in `src/unilab/structured_configs.py`
(`SACConfig`); `src/unilab/conf/sac/config.yaml` carries the same values unless noted.
Every field can be overridden on the command line:

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.gamma=0.99 \
  algo.algo_params.alpha_init=0.1
```

### Training efficiency

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 4096 | Parallel collector environments |
| `algo.batch_size` | int | 8192 | Learner batch per update |
| `algo.replay_buffer_n` | int | 512 | Replay capacity in env-vector sweeps (rows = n × num_envs) |
| `algo.updates_per_step` | int | 4 | Learner updates per env step |
| `algo.learning_starts` | int | 1 | Env-vector sweeps to collect before learner updates start |
| `algo.policy_frequency` | int | 4 | One actor/target update every N learner updates |
| `algo.max_iterations` | int | 500 | Number of training iterations |
| `algo.save_interval` | int | 500 | Checkpoint save interval in iterations |
| `algo.algo_params.use_compile` | bool | true | torch.compile on the learner update |
| `algo.seed` | int | 1 | Random seed |
| `training.use_amp` | bool | true | bf16 autocast on the learner update path |
| `training.log_interval` | int | 1 | TensorBoard/W&B scalar reporting interval in iterations |

### Policy performance

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.gamma` | float | 0.97 | Discount factor |
| `algo.tau` | float | 0.125 | Target-critic EMA coefficient |
| `algo.actor_lr` | float | 3e-4 | Actor Adam learning rate |
| `algo.critic_lr` | float | 3e-4 | Critic Adam learning rate |
| `algo.actor_hidden_dim` | int | 512 | Actor network width |
| `algo.critic_hidden_dim` | int | 768 | Critic network width |
| `algo.num_atoms` | int | 101 | Atoms of the distributional (C51) critic |
| `algo.obs_normalization` | bool | false | Normalize observations |
| `algo.use_layer_norm` | bool | true | LayerNorm in the networks |
| `algo.algo_params.alpha_lr` | float | 3e-4 | Entropy temperature (alpha) learning rate |
| `algo.algo_params.alpha_init` | float | 0.01 | Initial entropy temperature |
| `algo.algo_params.target_entropy_ratio` | float | 0.0 | Target entropy = −action_dim × ratio (0.0 → target 0) |
| `algo.algo_params.max_grad_norm` | float | 0.0 | Gradient norm clip; 0 disables clipping |
| `algo.algo_params.amp_dtype` | str | auto | AMP dtype under `training.use_amp` |

### Training runtime (`training:` section)

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `training.no_play` | bool | false | Skip policy playback after training |
| `training.play_only` | bool | false | Skip training and only play a checkpoint |
| `training.export_onnx` | bool | true | Export `policy.onnx` during playback |
| `training.cuda_process_sharing` | str / null | null | null = independent CUDA contexts; `mps` requests CUDA MPS (user-managed daemon via `uni-cumps`) |
| `training.dp_gradient_wire_dtype` | str | float32 | Gradient wire dtype for multi-GPU averaging; `bfloat16` compresses |
| `training.nan_guard.enabled` | bool | true | Detect non-finite tensors and dump diagnostics |
| `training.nan_guard.check_interval` | int | 32 | Env steps between NaN checks |
| `training.defer_finite_validation` | bool | true | Batch finite checks into a periodic cycle |
| `training.finite_validation_interval` | int | 32 | Steps per batched finite-check cycle |
| `training.collector_tensor_device` | str | auto | Device of collector-side tensors; `auto` follows the learner |
| `training.env_steps_per_sync` | int | 1 | Env steps per collection sync |
| `training.inference_slot_capacity` | int | 1 | In-flight learner-inference requests; 1 keeps inference serialized |
| `training.collector_metrics_interval` | int | 32 | Collector steps between metric flushes |
| `training.replay_ingress_depth` | int | 2 | Replay-ingress chunks allowed in flight |
| `training.replay_ingress_slot_rows` | int / null | null | Rows per ingress chunk; null = one chunk per collector vector |
| `training.replay_prefetch_mode` | str | one_tick | Replay prefetch strategy |
| `training.trace_enabled` | bool | false | Chrome-trace profiling of the off-policy runtime |
| `training.trace_output_dir` | str / null | null | Trace output directory |
| `training.trace_thread_time` | bool | false | Include thread time in traces |
| `training.trace_cuda_events` | bool | true | Record CUDA events in traces |
| `training.nvtx_profile_ranges` | bool | false | Emit NVTX ranges for external profilers |
| `training.torch_threads.enabled` | bool | true | Pin torch CPU thread counts per process role |
| `training.torch_threads.learner_num_threads` | int / str | auto | Learner intra-op threads; `auto` resolves from the host CPU count |
| `training.torch_threads.collector_num_threads` | int / str | auto | Collector intra-op threads |
| `training.dp_collector_cpu_ids` | list / null | null | Explicit CPU ids per rank; null auto-partitions `cpu_count // world_size` |
| `training.logger` | str | tensorboard | Scalar logging backend (`tensorboard` or `wandb`) |

### Reward term weights

Reward terms live in the task owner YAML under `reward:` — one block per term with
`func`, `weight`, and `params` (example:
`src/unilab/conf/sac/task/g1_walk_flat/base.yaml`). Retune a term by editing that
owner YAML or by overriding its weight on the command line:

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  reward.tracking_lin_vel.weight=2.5
```

Term names and default weights are task-specific; check the `reward:` block of the owner
YAML for the task you train (`src/unilab/conf/sac/task/<task>/base.yaml`).
