# W&B and TensorBoard

Training configs default to TensorBoard through `training.logger=tensorboard`.
Set `training.logger=wandb` to use Weights & Biases. The shared W&B fields live
in the training config blocks, including `wandb_project`, `wandb_entity`,
`wandb_group`, `wandb_name`, `wandb_tags`, `wandb_notes`, and `wandb_mode`.

## Examples

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco training.logger=tensorboard

uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

`ExperimentTracker` in `src/unilab/training/experiment.py` writes
`run_config.json` and `run_summary.json` in the run directory. For RSL-RL PPO,
the script also patches the RSL-RL W&B writer when `training.logger=wandb`.
