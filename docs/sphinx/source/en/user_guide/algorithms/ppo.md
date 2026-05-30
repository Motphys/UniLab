# PPO

PPO is the default synchronous on-policy training path. It uses
`scripts/train_rsl_rl.py`, composes from `conf/ppo/config.yaml`, and runs the
RSL-RL adapter code in `src/unilab/algos/torch/rsl_rl_ppo.py` and
`src/unilab/training/rsl_rl.py`.

## Quick Start

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix training.no_play=true
```

## Common Overrides

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

`algo.load_run` and `algo.checkpoint` select checkpoints for resume or playback:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

Logs are grouped by `algo.algo_log_name`; the default in `conf/ppo/config.yaml`
is `rsl_rl_ppo`.
