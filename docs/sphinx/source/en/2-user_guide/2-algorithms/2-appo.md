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

> The axis is schematic (relative, not real-ms). The collector subprocess produces rollouts through the 4-slot ring buffer in parallel with the learner, so **Collector Wait ≈ 0** in steady state. `perf/iter_ms` counts only this learner loop (it includes Collector Wait but not the collector's parallel rollout compute); the red Weight Sync marks the end of the iteration when fresh weights are published to the collector. Field meanings are on the [logging page](../1-training/3-logging.md).

## Key Fields

- `algo.steps_per_env`: rollout length per environment.
- `training.replay_queue_size`: learner-side cache depth.
- `training.collector_device`: collector device; defaults to following the learner.
- `algo.save_interval`: checkpoint save interval.

The default log root is `logs/appo/<task>/`, from `algo.algo_log_name=appo`
in `conf/appo/config.yaml`.
