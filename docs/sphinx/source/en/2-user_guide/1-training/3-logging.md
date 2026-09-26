# Logging

The terminal panel and TensorBoard / W&B are not separate logging systems. An
algorithm runner submits each metric once to the same training logger. The terminal
refreshes on a fixed 2 Hz clock and shows a two-second sliding average;
`training.logger=tensorboard` (the default) or `training.logger=wandb` selects the
persistent backend, which keeps each unsmoothed iteration value.

This page first covers the log directory shared by all algorithms and the
Manager-Based reward metric contract, then documents the off-policy terminal used by
SAC / FlashSAC / WarpSAC and APPO. Persisted backend tags follow the canonical schema
owned by `uni_rl.logging.metric_schema`: millisecond timing fields end in `_ms`,
while `Perf/learning_time`, `Perf/collection_time`, and `Perf/iteration_time` are
seconds. A few terminal fields (`Other`, phase percentages, collector cycle total,
`Rows/s`) are derived in memory for the display and are intentionally not persisted.

## Log Directory and Backend

Run training with the default TensorBoard backend:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

Run directories default to `logs/<algo.algo_log_name>/<task>/` unless the selected
stack overrides `training.log_root` or `training.log_dir`:

| Algorithm | Log root | `algo_log_name` source |
| --- | --- | --- |
| PPO | `logs/rsl_rl_ppo/<task>/` | `src/unilab/conf/ppo/config.yaml` |
| APPO | `logs/appo/<task>/` | `src/unilab/conf/appo/config.yaml` |
| SAC | `logs/fast_sac/<task>/` | `src/unilab/conf/sac/config.yaml` |
| FlashSAC | `logs/flash_sac/<task>/` | `src/unilab/conf/flashsac/config.yaml` |
| WarpSAC | `logs/warp_sac/<task>/` | `src/unilab/conf/warpsac/config.yaml` |

A run directory is named `YYYY-MM-DD_HH-MM-SS_<sim_backend>`, for example
`2026-03-09_18-30-00_mujoco`. Common artifacts include `run_config.json`,
`run_summary.json`, checkpoints, and `play_video.mp4` when that run produced one.

Enable W&B with:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

Shared fields include `training.wandb_project`, `training.wandb_entity`,
`training.wandb_group`, `training.wandb_name`, `training.wandb_tags`,
`training.wandb_notes`, and `training.wandb_mode`. `ExperimentTracker` writes
`run_config.json` and `run_summary.json`; RSL-RL PPO also connects its writer in W&B
mode. A MuJoCo `play_video.mp4` is uploaded when the run produces one.

## Manager-Based Reward Metrics

Manager-Based environments place structured metrics in `info["log"]`. The RSL-RL PPO
logger forwards slash-prefixed entries in that dictionary to TensorBoard / W&B and also
writes its own native aggregate training metrics. The reward-related keys have the
following contract:

| Key | Owner | Meaning |
| --- | --- | --- |
| `reward/<term>` | UniLab `RewardManager` | Per-term diagnostic for the current reward computation: the mean across environments of `raw_value * weight`, before `dt` scaling |
| `Train/mean_reward` | Native RSL-RL PPO logger | Aggregate mean completed-episode return; use this for overall training progress |
| `Train/mean_episode_length` | Native RSL-RL PPO logger | Aggregate mean completed-episode length |
| `Episode_Termination/<term>` | UniLab `TerminationManager` | Count of reset environments for which that termination term fired; this is a distinct diagnostic and is not a reward |
| `Episode_Reward/<term>` | Removed | Historical normalized cumulative per-term value; it is intentionally no longer emitted because it duplicated `reward/<term>` with a different meaning |

Historical TensorBoard event files are immutable, so runs made before this change can
still display `Episode_Reward/<term>`. New Manager-Based runs should not emit it.

## Off-policy Terminal

The bottom of the terminal has three columns:

- `Learner (Iter Wall)` is one learner-main-thread iteration; every percentage uses
  `Iter Wall` as its denominator.
- `Collector (own clock)` is measured in the collector subprocess and runs in
  parallel with the learner.
- `System` contains the buffer size, timeout rate, environment count, and per-rank
  batch size.

The panel-border title reports `GPUs N`. In multi-GPU training, only rank 0 owns the
terminal and persistent logger. Learner metrics and timings are averaged across
ranks first and then across the terminal's two-second window. `Steps/s` and
`Rows/s` are exceptions to the rank average: per-rank collector step rates and
learner replay-row rates are summed, so both fields are total job throughput. Only
`Steps/s` is persisted (as `Perf/total_fps`); `Rows/s` is terminal-only. The
`Avg 2s (n=...)` header field gives the number of already rank-reduced learner
samples in the current time window.

Do not add times across the learner and collector columns. Only the learner rows from
`Collector Wait` through `Other` are mutually exclusive main-thread phases. The
implementation fills unnamed intervals with `Other = max(Iter Wall - accounted, 0)`.
Normally the rows sum to `Iter Wall`, so their displayed percentages sum to about
100% (per-row rounding can introduce a small difference). The terminal no longer
hides applicable phases below a 1% threshold: a zero row is kept so it can be matched
directly with TensorBoard / W&B. Algorithm-specific phases neither occupy terminal
rows nor get persisted for other algorithms; for example, `Replay Stage` and
`Weight Publish` exist only for APPO.

### Learner Main Timeline

| Terminal field | TensorBoard / W&B key | Path | Meaning |
| --- | --- | --- | --- |
| Collector Wait | `Perf/learner_collector_wait_ms` | All | Learner-main-thread wait to reach this iteration's update boundary; SAC-like paths serve inference requests and continue until replay is ready and the `env_steps_per_sync` tick count is met, while APPO waits for a rollout in the ring |
| Inference | `Perf/learner_inference_ms` | Learner-owned inference | Total wall time for observation H2D, actor forward, and action D2H; nested details are listed below |
| Collector Release | `Perf/learner_collector_release_ms` | Learner-owned inference | Publishing the action response token; normally short, while blocking means the response queue has not drained |
| Replay Batch Wait | `Perf/learner_replay_batch_wait_ms` | Device replay | Waiting for the prefetched device batch to finish ingress commit and gather; near zero on a prefetch hit |
| Replay Stage | `Perf/learner_replay_stage_ms` | APPO | Sequentially materializing newly arrived NumPy rollouts from the ring into the learner staging pool; this is an exclusive learner-main-thread phase |
| Replay Sample | `Perf/learner_replay_sample_ms` | All | Acquiring the ready batch; usually hot/cold swap plus views on CUDA, with possible slot-event waiting on MPS |
| Learning | `Perf/learning_time` (seconds) | All | Learner update-phase wall time |
| Weight Publish | `Perf/learner_weight_publish_ms` | APPO | Writing fresh actor / critic weights to shared memory |
| Other | terminal-only | All | Residual after subtracting the phases above from `Iter Wall`, including metrics drain, reward stats, and loop bookkeeping |
| Iter Wall | `Perf/iteration_time` (seconds) | All | Wall time from learner-loop iteration start through update completion; always shown as 100%; emitted only when the runner measured the complete iteration wall time |

`Other` and the per-row percentages are derived in memory for the terminal and are
not persisted as backend charts. Accounted time contains only the mutually exclusive
main-thread phases above, never nested or background work. Normally
`accounted + other = 100%`. If a clock anomaly or a future overlapping timer makes
`accounted > Iter Wall`, `Other` is clamped to zero and accounted exceeding 100% is a
contract violation to fix, not an interpretable parallel-work percentage.

### Learner Nested and Background Diagnostics

The following keys are persistent-backend diagnostics, not additional `Iter Wall`
slices:

| TensorBoard / W&B key | Parent or execution thread | Meaning |
| --- | --- | --- |
| `Perf/learner_inference_h2d_ms` | Child of `Inference` | Observation copy from the shared CPU slot to the learner device |
| `Perf/learner_inference_forward_ms` | Child of `Inference` | `learner.actor` inference, including the current device synchronization |
| `Perf/learner_inference_d2h_ms` | Child of `Inference` | Action copy into the shared CPU slot |
| `Perf/replay_ingress_h2d_submit_ms` | Replay ingress; CUDA daemon or MPS learner thread | CPU-side duration of the latest transition-span submission into the authoritative device ring; it can occur inside any main phase and must not be added to learner percentages |

The three inference details should approximately compose `Inference`; small gaps come
from Python work between timers. `Replay H2D Submit` overlaps the learner timeline:
CUDA submits non-blocking copies from the `replay_gpu_resident_sync` daemon, while MPS
advances them from existing learner calls. Use a trace for the real asynchronous GPU
copy / gather interval instead of treating submit wall time as extra iteration share.

### Tags in Existing Runs

Historical TensorBoard event files are not rewritten. Runs made with unilab-rl 1.4.1
or later use the canonical tags below (source of truth:
`uni_rl.logging.metric_schema`, with the full migration table in the unilab_rl
repo's `docs/metrics.md`), while an older run continues to show its retired names:

| Old tag | New tag | Note |
| --- | --- | --- |
| `perf/steps_per_sec` | `Perf/total_fps` | Aggregate collector env-step throughput (cross-rank sum in DP) |
| `reward/mean`, `reward/mean_ep100` | `Train/mean_reward` | Collector's latest 100-episode mean return; the runner's 10-report smoothing is checkpoint state only |
| `episode/timeout_rate` | `Episode/timeout_rate` | Omitted until the first completed episode |
| `perf/iter_ms` | `Perf/iteration_time` | Milliseconds → seconds; emitted only for a measured complete iteration wall time |
| `timing/learner_train_ms` | `Perf/learning_time` | Milliseconds → seconds |
| `timing/learner_<phase>_ms` | `Perf/learner_<phase>_ms` | Same millisecond phase timings, canonical namespace |
| `timing/inference_total_ms`, `timing/inference_{h2d,forward,d2h}_ms` | `Perf/learner_inference_ms`, `Perf/learner_inference_{h2d,forward,d2h}_ms` | Pre-canonical names of the same fields |
| `timing/learner_incremental_h2d_ms` (SAC-like), `timing/replay_ingress_h2d_submit_ms` | `Perf/replay_ingress_h2d_submit_ms` | Potentially parallel submit diagnostic, not a learner main phase |
| `timing/learner_incremental_h2d_ms` (APPO) | `Perf/learner_replay_stage_ms` | Synchronous staging that is part of `Iter Wall` |
| `timing/collector_<phase>_ms` | `Perf/collector_<phase>_ms` | Same millisecond collector phases, canonical namespace |
| `timing/collector_inference_wait_ms` | `Perf/collector_learner_action_wait_ms` | The wait can include the remaining learner update, not only inference latency |
| `timing/collector_rollout_ms` | `Perf/collection_time` | Milliseconds → seconds; APPO's whole-rollout wall time |
| `train/dp_sync_time` | `Perf/dp_gradient_sync_ms_per_rank` | Seconds → milliseconds; explicitly per rank |
| `train/dp_gradient_sync_calls` | `Perf/dp_gradient_sync_calls_per_rank` | Explicitly per rank |

Retired without a replacement tag: `perf/effective_samples_per_sec` /
`perf/learner_samples_per_sec` (derive replay rows from the run configuration and
`Perf/iteration_time`), `perf/collector_active_steps_per_sec` (derive from collector
timing and the run configuration), `perf/collector_cycle_ms`, `perf/learner_*_pct`,
`timing/learner_other_ms`, and `perf/learner_pipeline_ms` — all of these are
derivable from the canonical fields above or are terminal-only derived values. The
extraction helper `scripts/benchmark/rl/extract_offpolicy_metrics.py` accepts both
schemas, preferring the canonical tag and scaling units to match.

### Collector Timeline

SAC / FlashSAC record four mutually exclusive hot-path phases per vectorized
env tick. Terminal percentages use the sum of these four phases (the collector
cycle), which is derived in memory and not persisted as a backend chart:

| Terminal field | TensorBoard / W&B key | Meaning |
| --- | --- | --- |
| Inference Request | `Perf/collector_inference_request_ms` | Publish observations / dones to the shared slot and notify the learner |
| Learner Action Wait | `Perf/collector_learner_action_wait_ms` | Barrier wall time from request publication until the learner publishes this tick's action |
| Env Step | `Perf/collector_env_step_ms` | `env.step()` wall time |
| Replay Write | `Perf/collector_replay_write_ms` | Transition post-processing, packing, and bounded-ingress write |

`Learner Action Wait` is deliberately not named “Inference Wait”: it is not pure
inference latency. If the collector finishes `Env Step + Replay Write` and submits its
next request while the learner is still updating, the value includes the remaining
update, a small scheduling part of the next learner `Collector Wait`, and the next
`Inference + Collector Release`. A long value therefore agrees with parallel
execution: it means the collector reached the next barrier before the learner.

The collector active-throughput diagnostic (`num_envs / (Inference Request +
Env Step + Replay Write)`, intentionally excluding `Learner Action Wait`) is no
longer persisted as a backend chart; it can be derived from the canonical collector
timing fields and the run configuration when needed. The terminal `Steps/s` field
instead reports total synchronized collector throughput, persisted as
`Perf/total_fps`. The indented Backend Step / Update State / Reset
Done rows are nested `Env Step` details. They do not enter the cycle sum, though their
displayed percentages use the same collector-cycle denominator.

APPO has a different collection contract and reports:

| Terminal field | TensorBoard / W&B key | Basis |
| --- | --- | --- |
| MLP Infer | `Perf/collector_mlp_infer_ms` | Per-step policy-inference EMA |
| Env Step | `Perf/collector_env_step_ms` | Single-`env.step()` EMA |
| Rollout Wall | `Perf/collection_time` (seconds) | Whole `steps_per_env` rollout wall time |

These are not one percentage breakdown: the first two are per-step EMAs and `Rollout
Wall` is a whole-rollout total, so the terminal shows milliseconds only. The backend
active-throughput diagnostic `(num_envs * steps_per_env) / Rollout Wall` is likewise
no longer persisted and can be derived from the fields above.

## FastSAC Dual Timeline

With the default `training.env_steps_per_sync=1`, this sequence starts at learner
iteration `k`. Horizontal messages are synchronization points; the `par` branches are
the actual overlap:

```{mermaid}
sequenceDiagram
    autonumber
    participant C as Collector (CPU / NumPy Env)
    participant I as Inference Slot + Queue
    participant L as Learner Main Thread
    participant D as Device Replay / CUDA Daemon

    Note over C,L: iteration k starts with request(t) already published
    C->>I: Inference Request(t)
    I-->>L: Collector Wait ends
    rect rgb(235, 245, 255)
        L->>I: Inference: observation H2D
        L->>L: learner.actor forward
        L->>I: action D2H
        L-->>C: Collector Release / response(t)
    end

    par Collector tick t
        C->>C: Env Step(t)
        C->>D: Replay Write(t)
        C->>I: Inference Request(t+1)
        Note over C,I: Learner Action Wait(t+1) starts here
    and Learner iteration k
        L->>D: Replay Batch Wait + Replay Sample(k)
        L->>L: Train(k): fixed updates_per_step
        Note over D,L: replay-ingress H2D / gather may overlap Train in background
    end

    I-->>L: iteration k+1 Collector Wait ends
    L->>I: Inference(t+1)
    L-->>C: response(t+1), ending Learner Action Wait
```

The important relationships are:

- Learner `Inference(t)` is the final part of collector `Learner Action Wait(t)`;
  they are not expected to have equal duration.
- `Env Step(t) + Replay Write(t)` overlaps learner replay work and `Train(k)`.
- The next request may arrive before `Train(k)` finishes, but the learner serves it
  only after the complete update. `Learner Action Wait(t+1)` can therefore be long
  while the next learner `Collector Wait` remains near zero.
- Inference does not overlap learner updates; each tick uses a complete
  `policy_version`.

## APPO Dual Timeline

The APPO collector continuously produces rollouts while the learner consumes complete
ring-buffer slots:

```{mermaid}
gantt
    title One APPO learner iteration (schematic scale)
    dateFormat x
    axisFormat %S

    section Collector (separate process)
    rollout N+1: MLP Infer + Env Step + other collection work :active, c0, 0, 18000

    section Ring Buffer
    rollout N ready   :milestone, r0, 0, 0
    rollout N+1 ready :milestone, r1, 18000, 18000

    section Learner (Iter Wall)
    Collector Wait          :done,   l0, 0, 1000
    Replay Stage            :        l1, 1000, 3000
    Replay Sample           :        l2, 3000, 4000
    Train                   :active, l3, 4000, 16000
    Weight Publish          :crit,   l4, 16000, 18000

    section Learner Total
    Iter Wall               :        l5, 0, 18000
```

Collector rollout and learner `Iter Wall` are independent, overlapping timelines and
must not be added. `Collector Wait` is near zero when the ring already contains a
slot. Collector `Rollout Wall` may be greater than, equal to, or less than one learner
`Iter Wall`, depending on ring backlog and relative throughput.

`Replay Stage` is the main-thread work that inserts every newly arrived slot into the
staging pool. `Replay Sample` then combines a training batch from that pool. They are
adjacent, additive learner phases, not background H2D diagnostics.

## Diagnosing Bottlenecks

| Observation | Direct meaning | Check first |
| --- | --- | --- |
| Learner `Collector Wait` is high | The collector's data / request is not ready when the learner reaches iteration start | Env step, transition post-processing, collector liveness, and IPC |
| Collector `Learner Action Wait` is high while learner `Collector Wait` is low | Collector reaches the next barrier first and waits for update completion plus inference | `Learning`, `updates_per_step`, and batch size; then inference details |
| Learner `Inference` is high | The learner-owned action path itself is slow | H2D / forward / D2H children |
| Learner `Replay Batch Wait` is high | Device-replay prefetch misses the consumption point | Ingress commit, side-stream gather, and GPU contention |
| Collector `Replay Write` is high | Bounded-ingress write or transition post-processing slows down | Exhausted ingress slots and delayed device commit |
| `Replay H2D Submit` is high but main phases are normal | Background submission diagnostic grew; it did not add the same wall time to the iteration | GPU copy, ingress commit, and overlap in Perfetto |

## Trace Options

Off-policy configs expose `training.trace_enabled`, `training.trace_output_dir`,
`training.trace_thread_time`, and `training.trace_cuda_events`. Scalar logs answer how
long each iteration took. Use the generated Perfetto timeline to determine whether
asynchronous H2D, device gather, collector, and learner work actually overlap.
