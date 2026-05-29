# Resume And Checkpoints

Checkpoint selection is controlled by algorithm-level fields. Use
`algo.load_run`, not `training.load_run`.

## Resume Training

Use a run id or `-1` for the latest run in the relevant log directory:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  algo.load_run=-1 \
  training.no_play=true

uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco \
  algo.load_run=2026-03-16_01-35-12_mujoco \
  training.no_play=true
```

## Replay A Checkpoint

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

Direct script playback uses `training.play_only=true`:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

Some script paths accept a checkpoint path through `algo.load_run`; the unified
CLI validates `--load-run` as a run id and does not accept path separators.

## Seeds

Training seed resolution is implemented in `src/unilab/training/seed.py`.
Algorithm configs currently carry `algo.seed`, and the helper records seed
metadata when experiment tracking is active.
