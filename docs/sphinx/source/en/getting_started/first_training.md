# First Training

This walkthrough starts a small PPO training job from a fresh checkout. It uses
the checked-in Go2 joystick owner YAML and keeps the run short enough for a
smoke test.

## Run A Smoke Job

Install dependencies first:

```bash
uv sync --extra motrix
```

Start a one-iteration MuJoCo run:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

The command composes `conf/ppo/config.yaml` with
`conf/ppo/task/go2_joystick_flat/mujoco.yaml`, builds the env through the
registry, and writes outputs under `logs/<algo.algo_log_name>/<task>/`.

## Try Motrix

Motrix is selected through the task owner, not by overriding
`training.sim_backend`:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

If Motrix is not installed, sync the extra with `uv sync --extra motrix`.

## Next Steps

- Replay a checkpoint with {doc}`evaluation_and_playback`.
- Learn the unified CLI in {doc}`../user_guide/training/cli_reference`.
- Read how Hydra owner YAMLs work in {doc}`../user_guide/training/hydra_config`.
