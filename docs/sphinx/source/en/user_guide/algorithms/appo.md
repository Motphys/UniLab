# APPO

APPO is UniLab's asynchronous PPO path. It uses `scripts/train_appo.py`,
`conf/appo/config.yaml`, and the runtime under `src/unilab/algos/torch/appo/`.
The config exposes `algo.steps_per_env`, `training.collector_device`, and
`training.replay_queue_size`; the algorithm config includes V-trace clipping
fields.

## Quick Start

```bash
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco
uv run scripts/train_appo.py task=g1_motion_tracking/motrix training.no_play=true
```

## Common Overrides

```bash
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.replay_queue_size=2
```

Playback and checkpoint selection use the same Hydra keys as PPO:

```bash
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

The default log family is `appo`, from `algo.algo_log_name` in
`conf/appo/config.yaml`.
