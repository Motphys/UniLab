# W&B 与 TensorBoard

语言: 简体中文

训练配置默认通过 `training.logger=tensorboard` 使用 TensorBoard。设置 `training.logger=wandb` 以使用 Weights & Biases。共享的 W&B 字段位于训练配置块中，包括 `wandb_project`、`wandb_entity`、`wandb_group`、`wandb_name`、`wandb_tags`、`wandb_notes` 和 `wandb_mode`。

## 示例

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.logger=tensorboard

uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  training.logger=wandb \
  training.wandb_project=unilab
```

`src/unilab/training/experiment.py` 中的 `ExperimentTracker` 会在运行目录下写入 `run_config.json` 和 `run_summary.json`。对于 RSL-RL PPO，脚本还会在 `training.logger=wandb` 时给 RSL-RL 的 W&B writer 打补丁。

## Navigation

- Index: [文档](0-index.md)
