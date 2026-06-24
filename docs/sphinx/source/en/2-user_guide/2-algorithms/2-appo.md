# APPO

APPO is UniLab's asynchronous PPO path. It uses `scripts/train_appo.py`,
`conf/appo/config.yaml`, and the runtime under `src/unilab/algos/torch/appo/`.
The config exposes `algo.steps_per_env`, `training.collector_device`, and
`training.replay_queue_size`; the algorithm config includes V-trace clipping
fields.

## Quick Start

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
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
sequenceDiagram
    participant C as Collector
    participant R as Ring Buffer
    participant L as Learner
    participant G as GPU
    loop Collect one rollout (steps_per_env steps)
        Note over C: mlp_infer_ms — per-step policy inference
        Note over C: env_step_total_ms — single env.step() time
    end
    C->>R: write rollout (Sync Collect)
    Note over L: Collector Wait — block until ring buffer has a new rollout
    R->>L: read available rollout (Rollouts Read / Available On Arrive)
    L->>G: stage into staging pool (Staging Pool sliding window)
    Note over L,G: H2D Copy — host-to-device batch copy
    L->>G: V-trace correction + PPO update (Appo/Updates Executed)
    Note over L,G: Train — pure SGD compute
    L->>C: write new weights to shared memory
    Note over L: Weight Sync — publish weights to the collector
    Note over L: Iter Wall — whole learner-iteration wall time (includes the above)
```

## Key Fields

- `algo.steps_per_env`: rollout length per environment.
- `training.replay_queue_size`: learner-side cache depth.
- `training.collector_device`: collector device; defaults to following the learner.
- `algo.save_interval`: checkpoint save interval.

The default log root is `logs/appo/<task>/`, from `algo.algo_log_name=appo`
in `conf/appo/config.yaml`.
