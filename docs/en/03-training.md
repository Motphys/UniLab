# Training Guide

Languages: English | [简体中文](../zh_CN/03-training.md) | [日本語](../ja/03-training.md) | [한국어](../ko/03-training.md)

This page covers training, playback, resume flow, Hydra overrides, and W&B.

## Pick An Entrypoint

| Goal | Entrypoint | Default log root |
|------|------------|------------------|
| PPO (RSL-RL / torch) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX / macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

## Start Training

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# PPO (MLX, Apple Silicon)
uv run python scripts/train_mlx_ppo.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# Off-policy
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# CLI overrides
uv run python scripts/train_offpolicy.py algo=sac task=g1_sac algo.num_envs=2048 algo.max_iterations=1000
```

By default, training scripts enter automatic playback after training finishes.

- `mujoco` exports `play_video.mp4`
- `motrix` renders in an interactive window
- `training.no_play=true` skips automatic playback

Run directories use the format `YYYY-MM-DD_HH-MM-SS_<sim_backend>`, for example `2026-03-09_18-30-00_mujoco`.

## Playback

```bash
# Play the latest result
uv run python scripts/train_rsl_rl.py task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# Play a specific run
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

## Resume Training

```bash
uv run python scripts/train_rsl_rl.py task=go2_joystick training.load_run="2024-02-04_12-00-00"
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

All training scripts are driven by Hydra config.

```bash
# Generic form
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# Common parameters
task=go1_joystick
algo=sac
training.play_only=true
training.no_play=true
training.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

Inspect the fully composed config with:

```bash
uv run python scripts/train_offpolicy.py --cfg job
```

## W&B

Set `training.logger=wandb` to enable automatic logging to Weights & Biases. Training scripts also write the following files into the local run directory:

- `run_config.json`
- `run_summary.json`

If the backend is `mujoco` and training produces `play_video.mp4`, the video is uploaded to the current W&B run as well.

```bash
# Basic usage
uv run python scripts/train_rsl_rl.py task=go1_joystick training.logger=wandb

# Shared project / entity
uv run python scripts/train_appo.py \
  task=go1_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# Group by task
uv run python scripts/train_offpolicy.py \
  algo=sac \
  task=go2_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=go2_joystick
```

Common fields:

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

Automatically recorded metadata includes the task, algorithm, backend, device, hardware information, git information, full config, total runtime, summary metrics, and the final playback video when available.

## Navigation

- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
