# Algorithms


This page only retains algorithm-level descriptions. For entry-point scripts and common CLI parameters, see {doc}`Training Guide <../getting_started/training>`.

## APPO

APPO is UniLab's asynchronous PPO implementation with V-trace importance-sampling correction. A collector subprocess handles CPU simulation, while the learner process handles GPU training; the two run in parallel through a ring-buffer pipeline.

### Core Features

| Feature | Description |
|------|------|
| Async multi-process | collector and learner run in parallel |
| V-trace IS correction | uses `pi_target / pi_behavior` to correct off-policy data |
| 4-slot ring buffer | up to 4 rollouts may be in flight simultaneously |
| Replay queue | learner-side queue caching rollouts pending consumption |
| Log directory | `logs/<algo.algo_log_name>/<task>/<timestamp>_<sim_backend>/` |

### Usage

```bash
# Default training
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco

# Specify number of envs and iterations
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco algo.num_envs=2048 algo.max_iterations=300

# Adjust replay queue depth
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.replay_queue_size=2

# Skip auto playback
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.play_only=true
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.play_only=true algo.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| Parameter | Default | Description |
|------|--------|------|
| `task` | `go1_joystick_flat/mujoco` | Single task config entry; internally defines both task + backend |
| `algo.max_iterations` | 150 | Max training iterations |
| `algo.num_envs` | 2048 | Number of parallel environments |
| `algo.steps_per_env` | 24 | Rollout length per env |
| `training.replay_queue_size` | 3 | learner-side rollout replay depth |
| `training.device` | auto-detect | learner device |
| `training.collector_device` | auto (follows `training.device`) | collector device |
| `training.logger` | `tensorboard` | Logging backend |
| `training.play_only` | false | Playback only |
| `training.no_play` | false | Skip auto playback |
| `algo.load_run` | `-1` | Run directory name or checkpoint path |
| `algo.save_interval` | 50 | Checkpoint save interval |

### APPO vs PPO

| Dimension | rsl-rl PPO | APPO |
|------|------------|------|
| Collection mode | Sync | Async |
| IS correction | None | V-trace |
| CPU / GPU utilization | Alternating full load | Simultaneous full load |
| Suited for | Sample-efficiency priority | Throughput priority |

## FastSAC And FastTD3

FastSAC and FastTD3 share the same async multi-process architecture, decoupling CPU simulation and GPU training through shared memory.

### Core Features

| Feature | Description |
|------|------|
| Async multi-process | collector and learner run independently |
| Unified shared memory | Zero-copy transfer using PyTorch shared tensors |
| Sync / async modes | Supports both default sync collection and async collection |
| Auto playback | Automatically enters playback after training |

### Usage

```bash
# Basic training
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco

# Async collection mode
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.no_sync_collection=true

# Skip auto playback
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco training.play_only=true algo.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| Parameter | Default | Description |
|------|--------|------|
| `algo` | `sac` | Algorithm selection |
| `task` | `sac/g1_walk_flat/mujoco` | Single task config entry; internally defines algo + task + backend |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | Max training iterations |
| `algo.num_envs` | 4096 | Number of parallel environments |
| `training.device` | auto-detect | learner device |
| `conf/*/task/...` | - | Sole owner config entry; reward/env/backend-specific algo are all changed here |
| `training.no_sync_collection` | false | Enables async collection |
| `training.env_steps_per_sync` | 1 | Collection steps per round in sync mode |
| `training.play_only` | false | Playback only |
| `training.no_play` | false | Skip auto playback |

## FlashSAC

FlashSAC is an off-policy algorithm based on FlashAttention-style Blocks. The actor uses a BatchNorm embedder + structured noise exploration, and the critic uses distributional Q (a C51 variant). It shares the same training entry point `train_offpolicy.py` with FastSAC, but the network architecture and forward interface differ.

### Usage

```bash
# Basic training
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco

# Skip auto playback
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.no_play=true
```

### Playback

```bash
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco training.play_only=true algo.load_run="2026-04-23_14-06-57_mujoco"
```

### Key Parameters

| Parameter | Default | Description |
|------|--------|------|
| `algo` | `flashsac` | Algorithm selection |
| `task` | `flashsac/g1_walk_flat/mujoco` | Sole owner config entry; reward/env/algo are all changed here |
| `algo.max_iterations` | 5000 | Max training iterations |
| `algo.num_envs` | 1024 | Number of parallel environments |
| `algo.tau` | 0.01 | Target network soft-update coefficient |
| `algo.algo_params.actor_num_blocks` | 2 | Actor FlashSAC block layers |
| `algo.algo_params.critic_num_blocks` | 2 | Critic FlashSAC block layers |
| `training.play_only` | false | Playback only |
| `training.no_play` | false | Skip auto playback |
