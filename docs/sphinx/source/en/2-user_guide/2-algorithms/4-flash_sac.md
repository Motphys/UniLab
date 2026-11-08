# FlashSAC

FlashSAC runs through `src/unilab/scripts/train_flashsac.py` in its own config tree.
Select it with `--algo flashsac`; defaults are inlined in
`src/unilab/conf/flashsac/config.yaml`, and the implementation lives under
`unilab.rl.algos.flash_sac`.

It shares the off-policy runner design with SAC, but does not use the
same default networks: the actor uses a block-based structure and the critic
uses a distributional (categorical) Q variant.

## Quick Start

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## Key Fields

For the off-policy playback path (`src/unilab/scripts/train_flashsac.py` / CLI `--algo flashsac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

## Learner acceleration controls

The default single-NVIDIA learner enables all three acceleration layers. They
remain separate configuration choices:

- `algo.algo_params.use_compile` controls CUDA Inductor compilation. Set it to
  `false` for the eager per-update compatibility path; this also disables the
  dependent whole-cycle graph.
- `algo.algo_params.compile_full_objectives` selects complete actor/critic
  objectives instead of the smaller loss-tensor compile regions. It does not
  control CUDA Graph capture.
- `algo.algo_params.use_whole_cycle_cuda_graph` controls learner-owned
  whole-cycle CUDA Graph capture. Set only this field to `false` to retain
  Inductor compilation without whole-cycle capture.

For example, run the eager debugging path with:

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  algo.algo_params.use_compile=false
```

FlashSAC requires synchronized collection and the same sole replay path as SAC:
bounded host ingress plus one complete replay ring on a CUDA or Apple
MPS learner device. CPU and XPU training are unsupported.

The log root is `logs/flash_sac/<task>/`.

## Checkpoint compatibility

Every FlashSAC checkpoint records an `actor_semantics` payload
(`schema_version`/`kind`/`builder`/`kwargs`/`obs_dim`/`action_dim`/
`action_transform`, written by `unilab.rl.algos.flash_sac.actor_spec`). Both
load paths — training resume and play — validate it fail-closed before any
weights are restored, for the built-in actor and for custom actors injected
through `algo.algo_params.actor_config` alike.

Pre-contract checkpoints (a missing or null `actor_semantics` entry) are no
longer supported for any actor, including the built-in actor: loading fails
closed with an error that states the minimum supported format. The only
recovery path is to retrain the policy with a current release. The supported
baseline starts with the first release after 1.3.5 that carries this change;
it is not backported to any 1.3.5 patch release.

## Configuration reference

Defaults come from the typed configs in `src/unilab/structured_configs.py`
(`FlashSACConfig`); `src/unilab/conf/flashsac/config.yaml` carries the same values unless
noted. Every field can be overridden on the command line:

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  algo.gamma=0.99 \
  algo.algo_params.temp_initial_value=0.1
```

### Training efficiency

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 1024 | Parallel collector environments |
| `algo.batch_size` | int | 2048 | Learner batch per update |
| `algo.replay_buffer_n` | int | 512 | Replay capacity in env-vector sweeps (rows = n × num_envs) |
| `algo.updates_per_step` | int | 2 | Learner updates per env step |
| `algo.learning_starts` | int | 98 | Env-vector sweeps to collect before learner updates start |
| `algo.policy_frequency` | int | 2 | One actor/temperature update every N learner updates |
| `algo.max_iterations` | int | 5000 | Number of training iterations |
| `algo.save_interval` | int | 1000 | Checkpoint save interval in iterations |
| `algo.algo_params.use_compile` | bool | true | CUDA Inductor compilation; `false` selects the eager path and also disables the whole-cycle graph |
| `algo.algo_params.compile_full_objectives` | bool | true | Compile complete actor/critic objectives instead of loss-tensor regions |
| `algo.algo_params.use_whole_cycle_cuda_graph` | bool | true | Learner-owned whole-cycle CUDA Graph capture |
| `algo.algo_params.dp_whole_cycle_cuda_graph` | bool | true | Whole-cycle capture on the data-parallel path; `false` disables DP capture only (field set in `config.yaml`) |
| `algo.seed` | int | 1 | Random seed |
| `training.use_amp` | bool | true | bf16 autocast on the learner update path |
| `training.log_interval` | int | 1 | TensorBoard/W&B scalar reporting interval in iterations |

### Policy performance

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.gamma` | float | 0.97 | Discount factor |
| `algo.tau` | float | 0.01 | Target-critic EMA coefficient |
| `algo.actor_lr` | float | 3e-4 | Fallback peak LR (used when `algo.algo_params.learning_rate_peak` ≤ 0) |
| `algo.critic_lr` | float | 3e-4 | Declared LR field; the effective schedule comes from `learning_rate_*` |
| `algo.actor_hidden_dim` | int | 128 | Actor network width |
| `algo.critic_hidden_dim` | int | 256 | Critic network width |
| `algo.num_atoms` | int | 101 | Atoms of the distributional critic |
| `algo.obs_normalization` | bool | false | Normalize observations |
| `algo.use_layer_norm` | bool | false | LayerNorm in the networks |
| `algo.policy_before_critic` | bool | false | Update actor (+temperature) before the critic in each update (field set in `config.yaml`) |
| `algo.algo_params.normalize_reward` | bool | true | Normalize rewards with a running return estimate |
| `algo.algo_params.normalized_g_max` | float | 5.0 | Clamp of the normalized return estimate |
| `algo.algo_params.actor_num_blocks` | int | 2 | Blocks in the block-based actor |
| `algo.algo_params.critic_num_blocks` | int | 2 | Blocks in the block-based critic |
| `algo.algo_params.actor_bc_alpha` | float | 0.0 | Behavior-cloning auxiliary loss weight; 0 disables it |
| `algo.algo_params.actor_noise_zeta_mu` | float | 2.0 | Actor noise schedule mu |
| `algo.algo_params.actor_noise_zeta_max` | int | 16 | Actor noise schedule cap |
| `algo.algo_params.critic_min_v` | float | -5.0 | Lower bound of the distributional critic support |
| `algo.algo_params.critic_max_v` | float | 5.0 | Upper bound of the distributional critic support |
| `algo.algo_params.temp_initial_value` | float | 0.01 | Initial entropy temperature |
| `algo.algo_params.temp_target_sigma` | float | 0.15 | Target sigma used by the temperature adaptation |
| `algo.algo_params.temp_target_entropy` | float / null | null | Explicit target entropy; null uses the built-in target |
| `algo.algo_params.learning_rate_init` | float | 3e-4 | Schedule start LR for actor/critic/temperature |
| `algo.algo_params.learning_rate_peak` | float | 3e-4 | Schedule peak LR |
| `algo.algo_params.learning_rate_end` | float | 1.5e-4 | Schedule end LR after decay |
| `algo.algo_params.learning_rate_warmup_steps` | int | 0 | Warmup steps from init to peak |
| `algo.algo_params.learning_rate_decay_steps` | int | 500000 | Decay steps from peak to end |
| `algo.algo_params.n_step` | int | 1 | N-step return length; only 1 is supported |
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
`src/unilab/conf/flashsac/task/g1_walk_flat/base.yaml`). Retune a term by editing that
owner YAML or by overriding its weight on the command line:

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  reward.tracking_lin_vel.weight=2.5
```

Term names and default weights are task-specific; check the `reward:` block of the owner
YAML for the task you train (`src/unilab/conf/flashsac/task/<task>/base.yaml`).
