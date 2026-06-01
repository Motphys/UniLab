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
