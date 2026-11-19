# PPO

PPO is the default synchronous on-policy training path. It uses
`src/unilab/scripts/train_rsl_rl.py`, composes from `src/unilab/conf/ppo/config.yaml`, and drives
upstream RSL-RL (rsl-rl-lib) directly: `unilab.rl.RslRlVecEnvAdapter`
adapts the env to RSL-RL's `VecEnv` contract, and RSL-RL's own `OnPolicyRunner`
trains `rsl_rl.algorithms:PPO`. The config follows the native RSL-RL v5 schema —
`algo.actor:` / `algo.critic:` model blocks (e.g. `class_name: rsl_rl.models.MLPModel`,
`hidden_dims`, `obs_normalization`) plus an `algo.algorithm:` block with
`class_name: rsl_rl.algorithms:PPO`.

## Quick Start

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.no_play=true
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

`CUDA_VISIBLE_DEVICES` (one entry per rank) enables RSL-RL's synchronous data-parallel PPO path:

```bash
export CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.no_play=true

uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  training.no_play=true
```

`CUDA_VISIBLE_DEVICES` is the sole GPU topology source. Parent entries may be
physical ordinals, UUIDs, or MIG tokens; torchrun remaps each worker to one
visible GPU, so every rank uses its local `cuda:0`. `training.device` remains
only an explicit single-device override and cannot replace rank-local
visibility.

For the IsaacGym, IsaacSim, and Genesis owners, the same topology is also
applied to the simulator environment. Genesis selects its process-wide session
before `gs.init`.

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
curriculum state after startup. Consequently, tasks with observation
normalization (`algo.actor.obs_normalization`, including the current Go2 flat
owner) keep rank-local statistics, and the checkpoint contains rank 0's copy.
This is the upstream RSL-RL distributed semantic, not global-rollout PPO.

The integrated launcher is single-node only. The repository has two-GPU MuJoCo
smoke coverage for `go2_joystick_flat` and `g1_motion_tracking`; this does not
claim multi-node or Motrix multi-GPU support. On the currently validated RTX
6000D host, the launcher inherits the production NCCL compatibility defaults
`NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1`; explicit environment values win.

## Configuration reference

Defaults come from the typed configs in `src/unilab/structured_configs.py`
(`PPOConfig`); `src/unilab/conf/ppo/config.yaml` carries the same values unless noted.
Every field can be overridden on the command line:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.algorithm.lam=0.92 \
  algo.algorithm.entropy_coef=0.005
```

### Training efficiency

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.num_envs` | int | 4096 | Parallel environments per rank (the go2_joystick_flat/mujoco owner sets 1024) |
| `algo.num_steps_per_env` | int | 24 | Rollout length per env; samples per iteration = num_envs × this |
| `algo.max_iterations` | int | 101 | Number of training iterations (the go2 owner sets 151) |
| `algo.save_interval` | int | 100 | Checkpoint save interval in iterations |
| `algo.algorithm.num_learning_epochs` | int | 5 | SGD epochs over each collected batch |
| `algo.algorithm.num_mini_batches` | int | 4 | Mini-batches per epoch |
| `algo.seed` | int | 1 | Base seed; rank `i` uses `seed + i` |
| `training.device` | str / null | null | Explicit single-device override; null = automatic placement |
| `training.log_interval` | int | 1 | TensorBoard/W&B scalar reporting interval in iterations |

### Policy performance

| Override key | Type | Default | Practical effect |
| --- | --- | --- | --- |
| `algo.algorithm.gamma` | float | 0.99 | Discount factor |
| `algo.algorithm.lam` | float | 0.95 | GAE lambda; lower trades bias for variance |
| `algo.algorithm.entropy_coef` | float | 0.01 | Entropy bonus coefficient (the go2 owner lowers it to 1e-3) |
| `algo.algorithm.learning_rate` | float | 1e-3 | Adam learning rate; initial value under the adaptive schedule (the go2 owner sets 3e-4) |
| `algo.algorithm.schedule` | str | adaptive | LR schedule; `adaptive` steers KL toward `desired_kl` |
| `algo.algorithm.desired_kl` | float | 0.01 | KL target of the adaptive LR schedule |
| `algo.algorithm.clip_param` | float | 0.2 | PPO surrogate clip range |
| `algo.algorithm.value_loss_coef` | float | 1.0 | Weight of the value loss term |
| `algo.algorithm.use_clipped_value_loss` | bool | true | Clip the value targets like the policy objective |
| `algo.algorithm.max_grad_norm` | float | 1.0 | Gradient norm clip |
| `algo.actor.hidden_dims` | list[int] | [512, 256, 128] | Policy MLP layer widths |
| `algo.actor.activation` | str | elu | Policy MLP activation |
| `algo.actor.obs_normalization` | bool | false | Normalize policy observations (the go2 owner enables it) |
| `algo.actor.distribution_cfg.init_std` | float | 1.0 | Initial Gaussian action std (the go2 owner sets 0.5) |
| `algo.critic.hidden_dims` | list[int] | [512, 256, 128] | Critic MLP layer widths |
| `algo.critic.obs_normalization` | bool | false | Normalize critic observations (the go2 owner enables it) |

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
| `training.num_timesteps` | int / null | null | Global sample budget; overrides the derived iteration count when set |

### Reward term weights

Reward terms live in the task owner YAML under `reward:` — one block per term with
`func`, `weight`, and `params` (example:
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`). Retune a term by editing that
owner YAML or by overriding its weight on the command line:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  reward.tracking_lin_vel.weight=1.5 \
  reward.action_rate.weight=-0.01
```

Term names and default weights are task-specific; check the `reward:` block of the owner
YAML for the task you train (`src/unilab/conf/ppo/task/<task>/base.yaml`).
