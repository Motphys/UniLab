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

For off-policy (SAC / TD3 / FlashSAC and APPO) the learner loop is reported as
named wall-clock phases, so `Train` can be read as a share of the full iteration
instead of being compared only against wait metrics.

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| Collector Wait | `timing/learner_collector_wait_ms` | Waiting for the collector to submit the next inference request and produce trainable data; APPO is ≈0 in steady state, while the synchronized off-policy path waits for its configured collection chunk |
| Replay Batch Wait | `timing/learner_replay_batch_wait_ms` | Waiting for the device batch prefetched at the end of the previous iteration (ingress commit + side-stream gather); ~0 on a prefetch hit, non-zero means the device replay pipeline is not keeping up with the learner |
| Replay Sample | `timing/learner_replay_sample_ms` | Consuming the already gathered hot device batch (hot/cold slot swap + view); ≈0 on CUDA, on MPS it includes the slot-event synchronization |
| Collector Release | `timing/learner_collector_release_ms` | The learner sending the release token to a synchronized collector; blocking here means the collector has not consumed the previous release |
| H2D Copy | `timing/learner_incremental_h2d_ms` | Submission cost of the incremental bounded-ingress copy into the authoritative device ring (CUDA: non-blocking submit on a daemon thread; MPS: blocking copy on the learner thread); on CUDA this work overlaps Train or already happens inside Collector Wait / Replay Batch Wait, so treat this row as a diagnostic that is not strictly additive with the other learner rows |
| Train | `timing/learner_train_ms` | Pure SGD compute |
| Weight Publish | `timing/learner_weight_publish_ms` | APPO only: publishing new actor weights to shared memory for its collector; zero for learner-owned off-policy inference |
| Iter Wall | `perf/iter_ms` | Whole-iteration wall time, not the sum of the components |

The terminal shows every learner row as right-aligned `ms` plus `% of Iter Wall`.
Backend logs include `perf/learner_train_pct`, `perf/learner_accounted_pct`,
`perf/learner_other_pct` and `timing/learner_other_ms`; the latter two are residual
diagnostics and are not shown as terminal rows. `perf/learner_pipeline_ms` = learner
inference + H2D + Train + APPO Weight Publish. The former `timing/learner_wait_ms` was renamed to
`timing/learner_collector_wait_ms`; the former `timing/learner_sync_coordination_ms`
and `timing/learner_weight_sync_ms` were renamed to
`timing/learner_collector_release_ms` and `timing/learner_weight_publish_ms`.

The collector process reports per-phase timings in the terminal Collector column and
TensorBoard `timing/collector_*`; each terminal row also shows the share of one
collection cycle (the sum of the rows below). SAC / TD3 / FlashSAC:

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| Inference Request | `timing/collector_inference_request_ms` | Publishing observations and dones to the shared inference slot and notifying the learner |
| Inference Barrier Wait | `timing/collector_inference_wait_ms` | Waiting for the learner-owned actor to publish this tick's action |
| Env Step | `timing/collector_env_step_ms` | Environment step |
| Replay Write | `timing/collector_replay_write_ms` | Packing transitions and writing them into the bounded replay ingress |
| Sync Idle | `timing/collector_sync_idle_ms` | Per-step episode bookkeeping and metrics reporting; excluded from `Collector/s` |

`Collector/s` is collector active throughput. For SAC / TD3 / FlashSAC it uses
`num_envs / (Inference Request + Env Step + Replay Write)` and excludes
`Inference Barrier Wait` and `Sync Idle`. The three indented child rows under Env Step (Backend Step / Update
State / Reset Done) are marked with tree connectors and share the same percentage
base as the other rows (one collection cycle).

### Reading CPU vs GPU load

The two sides of the pipeline expose matching wait signals; use them to tell which
side is the bottleneck:

| Observation | Meaning | Where to look |
| --- | --- | --- |
| Learner Collector Wait % is high | Learner is waiting on the collector's CPU environment and transition path | Scale `num_envs` or reduce environment/transition overhead |
| Learner Replay Batch Wait is high | The device replay pipeline (ingress commit + gather) is not keeping up with learner consumption | GPU is saturated or the side stream is queued behind training kernels |
| Collector Inference Barrier Wait is high | Collector waits for learner inference or the learner update phase | Inspect inference latency and reduce `updates_per_step` / batch size |
| Collector Replay Write grows | Writing into the bounded ingress slows down, e.g. slots exhausted waiting for device commits | Same direction as Replay Batch Wait: the device consumption side is behind |

APPO uses a ring buffer; the collector reports two **per-step** EMAs plus one **whole-rollout** total:

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| MLP Infer | `timing/collector_mlp_infer_ms` | EMA of per-step policy inference (**per step**) |
| Env Step | `timing/collector_env_step_ms` | EMA of a single `env.step()` (**per step**) |
| Rollout | `timing/collector_rollout_ms` | EMA of the real wall-clock time to produce **one full rollout** (`steps_per_env` steps); shown last in the column as the total |

For APPO, `Collector/s` uses `(num_envs * steps_per_env) / Rollout`.

> Rollout ≈ `steps_per_env` × (MLP Infer + Env Step) + untimed per-step overhead (e.g. the timeout-bootstrap critic forward, obs processing). It and the learner's Collector Wait are **two independent-timeline views**: collection overlaps the learner's compute, so Collector Wait (the time the learner is actually blocked) is normally **smaller** than Rollout, and the two are not meant to reconcile exactly. To see "how much of this iteration waits on the collector," read the percentage on the Collector Wait row (= Collector Wait / Iter Wall). The same percentage format is used for every learner row. The former `env_step_total_ms` (`timing/collector_env_step_total_ms`) is renamed to `Env Step` (`timing/collector_env_step_ms`).

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
    Weight Publish → collector            :crit,   l3, 28000, 30000

    section Iter Wall
    perf/iter_ms (learner loop only)      :        l4, 12000, 30000
```

> The axis is schematic (relative, not real-ms). The collector subprocess produces rollouts through the 4-slot ring buffer in parallel with the learner, so **Collector Wait ≈ 0** in steady state. `perf/iter_ms` counts only this learner loop (it includes Collector Wait but not the collector's parallel rollout compute); the red Weight Publish marks the end of the iteration when fresh weights are published to the collector.

All off-policy terminal views use the same value formatting. Replay Batch Wait is
shown only when a single-device/double-buffer prefetch miss is non-zero. Collector
Release is shown for the synchronized learner-owned inference path.
