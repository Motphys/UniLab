# Algorithms

Languages: English | [简体中文](../zh_CN/04-algorithms.md) | [日本語](../ja/04-algorithms.md) | [한국어](../ko/04-algorithms.md)

This page keeps only algorithm-level notes. For entrypoints and shared CLI parameters, see [03-training.md](03-training.md).

## APPO

APPO is UniLab's asynchronous PPO implementation with V-trace importance-sampling correction. Collector subprocesses handle CPU simulation, while the learner process handles GPU training through a ring-buffered pipeline.

### Core Features

| Feature | Description |
|---------|-------------|
| Async multiprocess | collectors and learner run in parallel |
| V-trace IS correction | correct off-policy data with `pi_target / pi_behavior` |
| 4-slot ring buffer | up to 4 rollouts can be in flight |
| Replay queue | learner-side queue for pending rollouts |
| Log directory | `logs/appo/<task>/<timestamp>_mujoco/` |

### Usage

```bash
# Default training
uv run python scripts/train_appo.py task=go1_joystick

# Set env count and iteration count
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# Tune replay queue depth
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# Skip automatic playback
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | `go2_joystick` | task config name |
| `algo.max_iterations` | 150 | maximum training iterations |
| `algo.num_envs` | 2048 | number of parallel envs |
| `algo.steps_per_env` | 24 | rollout length per env |
| `training.replay_queue_size` | 3 | learner-side rollout replay depth |
| `training.device` | auto-detected | learner device |
| `training.collector_device` | `cpu` | collector device |
| `training.logger` | `tensorboard` | logging backend |
| `training.play_only` | false | playback only |
| `training.no_play` | false | skip automatic playback |
| `training.load_run` | `-1` | run directory name or checkpoint path |
| `algo.save_interval` | 50 | checkpoint save interval |

### APPO vs PPO

| Dimension | rsl-rl PPO | APPO |
|-----------|------------|------|
| Collection mode | synchronous | asynchronous |
| IS correction | none | V-trace |
| CPU / GPU utilization | alternating saturation | concurrent saturation |
| Best fit | sample efficiency first | throughput first |

## FastSAC And FastTD3

FastSAC and FastTD3 use the same asynchronous multiprocess architecture to decouple CPU simulation from GPU training through shared memory.

### Core Features

| Feature | Description |
|---------|-------------|
| Async multiprocess | collectors and learner run independently |
| Unified shared memory | zero-copy transport through PyTorch shared tensors |
| Sync / async modes | support both default synchronized collection and async collection |
| Automatic playback | playback runs automatically after training |

### Usage

```bash
# Basic training
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# Async collection mode
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# Skip automatic playback
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algo` | `sac` | algorithm selection |
| `task` | `go1_joystick` | task config name |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | maximum training iterations |
| `algo.num_envs` | 4096 | number of parallel envs |
| `training.device` | auto-detected | learner device |
| `training.sim_backend` | `mujoco` | simulation backend |
| `training.no_sync_collection` | false | enable async collection |
| `training.env_steps_per_sync` | 1 | steps collected per sync cycle |
| `training.play_only` | false | playback only |
| `training.no_play` | false | skip automatic playback |

## Navigation

- Previous: [Training Guide](03-training.md)
- Next: [G1 Motion Tracking](05-g1-motion-tracking.md)
