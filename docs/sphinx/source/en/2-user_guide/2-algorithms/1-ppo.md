# PPO

PPO is the default synchronous on-policy training path. It uses
`scripts/train_rsl_rl.py`, composes from `conf/ppo/config.yaml`, and runs the
RSL-RL adapter code in `src/unilab/algos/torch/rsl_rl_ppo.py` and
`src/unilab/training/rsl_rl.py`.

## Quick Start

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## Common Overrides

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

Use `uv run eval` for checkpoint playback:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Logs are grouped by `algo.algo_log_name`; the default in `conf/ppo/config.yaml`
is `rsl_rl_ppo`.
