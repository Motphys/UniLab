# Logging

Training configs default to TensorBoard with `training.logger=tensorboard`.
Set `training.logger=wandb` to enable Weights & Biases integration.

## TensorBoard

Run any training command with the default logger:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

Run directories are created under `logs/<algo.algo_log_name>/<task>/` unless
`training.log_root` or `training.log_dir` is overridden by the selected stack.

### Log Roots Per Algorithm

`algo_log_name` is set by each stack's config and resolves to a concrete root:

| Algorithm | Log Root | `algo_log_name` Source |
| --- | --- | --- |
| PPO | `logs/rsl_rl_ppo/<task>/` | `conf/ppo/config.yaml` |
| MLX PPO | `logs/mlx_rl_train/<task>/` | `conf/ppo/config_mlx.yaml` |
| APPO | `logs/appo/<task>/` | `conf/appo/config.yaml` |
| SAC | `logs/fast_sac/<task>/` | `conf/offpolicy/algo/sac.yaml` |
| FlashSAC | `logs/flash_sac/<task>/` | `conf/offpolicy/algo/flashsac.yaml` |
| TD3 | `logs/fast_td3/<task>/` | `conf/offpolicy/algo/td3.yaml` |

### Run Directory Naming

A single run directory is named with a UTC-local timestamp plus the simulation
backend:

```text
YYYY-MM-DD_HH-MM-SS_<sim_backend>
```

For example, `2026-03-09_18-30-00_mujoco`. Common local artifacts written into a
run directory are:

- `run_config.json`
- `run_summary.json`
- checkpoint files
- `play_video.mp4` (MuJoCo, when that run produced a playback video)

## Weights & Biases

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

Supported shared W&B fields are declared in the training config blocks:

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode`

`src/unilab/training/experiment.py` writes `run_config.json` and
`run_summary.json` in the run directory. RSL-RL PPO also patches the RSL-RL W&B
writer when `training.logger=wandb`. When the backend is MuJoCo and a run
produces `play_video.mp4`, that video is uploaded to the W&B run.

## Trace Options

The off-policy config exposes trace fields such as
`training.trace_enabled`, `training.trace_output_dir`,
`training.trace_thread_time`, and `training.trace_cuda_events`.

## Off-Policy Timing Fields

For off-policy (SAC / TD3 / FlashSAC and APPO) the learner wait is split into four
independent components, reported separately and never merged.

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | Waiting for the collector to produce new data; excludes barrier, H2D and logger refresh |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | Waiting for a replay pack / H2D batch to become ready; ~0 on a prefetch hit |
| Rank Barrier | `timing/learner_rank_barrier_ms` | Multi-GPU `dist.barrier()` (initial + final) total |
| Sync Coordination | `timing/learner_sync_coordination_ms` | Synchronous-collection handshake; 0 when not in sync collection |
| H2D Copy | `timing/learner_incremental_h2d_ms` | Host-to-device batch copy |
| Train | `timing/learner_train_ms` | Pure SGD compute, excluding param sync and barrier |
| Param Sync | `timing/learner_param_sync_ms` | Multi-GPU local-SGD parameter averaging |
| Weight Sync | `timing/learner_weight_sync_ms` | Publishing new weights to the collector |
| Iter Wall | `perf/iter_ms` | Whole-iteration wall time, not the sum of the components |

Single GPU shows only Collector Wait, H2D Copy, Train, Weight Sync and Iter Wall;
Rank Barrier, Sync Coordination and Param Sync appear only on multi-GPU and are
recorded on rank 0 only. `perf/learner_pipeline_ms` = H2D + Train + Param Sync +
Weight Sync. The former `timing/learner_wait_ms` was renamed to
`timing/learner_collector_wait_ms`.

The collector process reports per-phase timings in the terminal Collector column and
TensorBoard `timing/collector_*`. SAC / TD3:

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| Weight Sync | `timing/collector_weight_sync_ms` | Pulling and loading new learner weights |
| Action Select | `timing/collector_action_select_ms` | Actor inference |
| Env Step | `timing/collector_env_step_ms` | Environment step |
| Replay | `timing/collector_replay_ms` | Replay buffer write and sample packing |
| Sync Coordination | `timing/collector_sync_coordination_ms` | Synchronous-collection handshake (signal learner, wait for learner) |

APPO uses a ring buffer; the collector reports only two fields, both **per-step** EMAs (not whole-rollout):

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| env_step_total_ms | `timing/collector_env_step_total_ms` | EMA of a single `env.step()` |
| mlp_infer_ms | `timing/collector_mlp_infer_ms` | EMA of per-step policy inference |

### Per-iteration sequence (APPO example)

The collector continuously produces rollouts through the ring buffer; each learner
iteration goes through the following timed components (the meaning is in parentheses):

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
    H2D Copy (ring → staging)             :        l1, 13000, 16000
    Train (V-trace + PPO SGD)             :active, l2, 16000, 28000
    Weight Sync → collector               :crit,   l3, 28000, 30000

    section Iter Wall
    perf/iter_ms (learner loop only)      :        l4, 12000, 30000
```

> The axis is schematic (relative, not real-ms). The collector subprocess produces rollouts through the 4-slot ring buffer in parallel with the learner, so **Collector Wait ≈ 0** in steady state. `perf/iter_ms` counts only this learner loop (it includes Collector Wait but not the collector's parallel rollout compute); the red Weight Sync marks the end of the iteration when fresh weights are published to the collector.

SAC / TD3 and multi-GPU paths additionally show Replay Batch Wait (waiting for a
replay pack / H2D batch), Rank Barrier (multi-GPU rank sync), Param Sync (multi-GPU
parameter averaging) and Sync Coordination (synchronous-collection handshake);
APPO single-GPU has none of these.
