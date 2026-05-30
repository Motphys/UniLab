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
writer when `training.logger=wandb`.

## Trace Options

The off-policy config exposes trace fields such as
`training.trace_enabled`, `training.trace_output_dir`,
`training.trace_thread_time`, and `training.trace_cuda_events`.
