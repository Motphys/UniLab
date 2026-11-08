# APPO

APPO is UniLab's asynchronous PPO path. It uses `src/unilab/scripts/train_appo.py`,
`src/unilab/conf/appo/config.yaml`, and the runtime under `unilab.rl.algos.appo`.
The config exposes `algo.steps_per_env`, `training.collector_device`, and
`training.replay_queue_size`; the algorithm config includes V-trace clipping
fields.

## Quick Start

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
```

## Common Overrides

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.replay_queue_size=2
```

Playback and checkpoint selection use `uv run eval`:

```bash
uv run eval --algo appo --task go2_joystick_flat --sim mujoco --load-run -1
```

## Runtime Model

- The collector runs CPU simulation while the learner runs GPU training.
- Rollouts are published into a replay queue that the learner consumes.
- APPO applies a V-trace importance-sampling correction, so its update
  semantics differ from synchronous PPO.
- The collector/learner pipeline is backed by a 4-slot ring buffer.

Per-iteration timing sequence (field meanings on the [logging page](../1-training/3-logging.md)):

```{mermaid}
gantt
    title Time inside one learner iteration (APPO)
    dateFormat x
    axisFormat %S

    section Collector (proc)
    rollout N · env interaction (mlp_infer + env_step) ×steps_per_env :active, c0, 0, 12000
    rollout N+1 (collected in parallel with learner)                  :active, c1, 13000, 30000

    section Ring Buffer (4 slots)
    rollout N ready    :milestone, r0, 12000, 12000
    rollout N+1 ready  :milestone, r1, 30000, 30000

    section Learner (GPU)
    Collector Wait (≈0 when buffer full)  :done,   l0, 12000, 13000
    Replay Stage (ring → staging)         :        l1, 13000, 15500
    Replay Sample (compose batch views)   :        l2, 15500, 16000
    Train (V-trace + PPO SGD)             :active, l3, 16000, 28000
    Weight Publish → collector            :crit,   l4, 28000, 30000

    section Iter Wall
    Perf/iteration_time (learner loop only)  :        l5, 12000, 30000
```

> The axis is schematic (relative, not real-ms). The collector subprocess produces rollouts through the 4-slot ring buffer in parallel with the learner, so **Collector Wait ≈ 0** in steady state. `Perf/iteration_time` counts only this learner loop (it includes Collector Wait but not the collector's parallel rollout compute); the red Weight Publish marks the end of the iteration when fresh weights are published to the collector. Field meanings are on the [logging page](../1-training/3-logging.md).

## Key Fields

- `algo.steps_per_env`: rollout length per environment.
- `training.replay_queue_size`: learner-side cache depth.
- `training.collector_device`: collector device; defaults to following the learner.
- `algo.save_interval`: checkpoint save interval.

The default log root is `logs/appo/<task>/`, from `algo.algo_log_name=appo`
in `src/unilab/conf/appo/config.yaml`.

## Configuration reference

Defaults come from the typed configs in `src/unilab/structured_configs.py`
(`APPOConfig`); `src/unilab/conf/appo/config.yaml` carries the same values unless noted.
Every field can be overridden on the command line:

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.algorithm.lam=0.92 \
  algo.algorithm.vtrace_clip_rho=1.2
```

### Training efficiency

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 2048 | Environments per collector |
| `algo.steps_per_env` | int | 24 | Rollout length per env in each published rollout |
| `algo.max_iterations` | int | 150 | Number of learner iterations |
| `algo.save_interval` | int | 50 | Checkpoint save interval in iterations |
| `algo.algorithm.num_learning_epochs` | int | 5 | SGD epochs over each consumed rollout |
| `algo.algorithm.num_mini_batches` | int | 4 | Mini-batches per epoch |
| `algo.algorithm.enable_compile` | bool | true | torch.compile on the learner update; the built-in `src/unilab/conf/appo/config.yaml` ships `false` |
| `algo.seed` | int | 1 | Random seed |
| `training.collector_device` | str / null | null | Collector device; null follows the learner |
| `training.replay_queue_size` | int / null | null | Learner-side replay cache depth |
| `training.log_interval` | int | 1 | TensorBoard/W&B scalar reporting interval in iterations |

### Policy performance

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.algorithm.gamma` | float | 0.99 | Discount factor |
| `algo.algorithm.lam` | float | 0.95 | GAE lambda; lower trades bias for variance |
| `algo.algorithm.entropy_coef` | float | 0.01 | Entropy bonus coefficient |
| `algo.algorithm.learning_rate` | float | 1e-3 | Adam learning rate; initial value under the adaptive schedule |
| `algo.algorithm.schedule` | str | adaptive | LR schedule; `adaptive` steers KL toward `desired_kl` |
| `algo.algorithm.desired_kl` | float | 0.01 | KL target of the adaptive LR schedule |
| `algo.algorithm.adaptive_kl_factor` | float | 1.2 | KL deadband: LR drops above `desired_kl` × this and rises below `desired_kl` ÷ this |
| `algo.algorithm.adaptive_lr_factor` | float | 1.1 | Multiplicative LR step of the adaptive schedule |
| `algo.algorithm.clip_param` | float | 0.2 | PPO surrogate clip range |
| `algo.algorithm.value_loss_coef` | float | 1.0 | Weight of the value loss term |
| `algo.algorithm.use_clipped_value_loss` | bool | true | Clip the value targets like the policy objective |
| `algo.algorithm.max_grad_norm` | float | 1.0 | Gradient norm clip |
| `algo.algorithm.optimizer` | str | adam | Optimizer for the learner update |
| `algo.algorithm.vtrace_clip_rho` | float | 1.0 | V-trace importance-weight clip ρ̄ |
| `algo.algorithm.vtrace_clip_c` | float | 1.0 | V-trace importance-weight clip c̄ |
| `algo.algorithm.tau` | float | 1.0 | Target-network EMA coefficient (1.0 = hard copy) |
| `algo.algorithm.target_update_freq` | int | 1 | Learner updates between target-network syncs |
| `algo.actor.hidden_dims` | list[int] | [512, 256, 128] | Policy MLP layer widths |
| `algo.actor.activation` | str | elu | Policy MLP activation |
| `algo.actor.distribution_cfg.init_std` | float | 1.0 | Initial Gaussian action std |
| `algo.critic.hidden_dims` | list[int] | [512, 256, 128] | Critic MLP layer widths |

### Training runtime (`training:` section)

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `training.no_play` | bool | false | Skip policy playback after training |
| `training.play_only` | bool | false | Skip training and only play a checkpoint |
| `training.tf32_matmul` | str | ieee | Learner FP32 matmul backend; `tf32` opts into tensor cores with automatic IEEE fallback |
| `training.nan_guard.enabled` | bool | true | Detect non-finite tensors and dump diagnostics |
| `training.nan_guard.check_interval` | int | 32 | Env steps between NaN checks |
| `training.defer_finite_validation` | bool | true | Batch finite checks into a periodic cycle |
| `training.finite_validation_interval` | int | 32 | Steps per batched finite-check cycle |
| `training.collector_tensor_device` | str | auto | Device of collector-side tensors; `auto` follows the learner |
| `training.logger` | str | tensorboard | Scalar logging backend (`tensorboard` or `wandb`) |

### Reward term weights

Reward terms live in the task owner YAML under `reward:` — one block per term with
`func`, `weight`, and `params` (example:
`src/unilab/conf/appo/task/go2_joystick_flat/base.yaml`). Retune a term by editing that
owner YAML or by overriding its weight on the command line:

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  reward.tracking_lin_vel.weight=1.5 \
  reward.action_rate.weight=-0.01
```

Term names and default weights are task-specific; check the `reward:` block of the owner
YAML for the task you train (`src/unilab/conf/appo/task/<task>/base.yaml`).
