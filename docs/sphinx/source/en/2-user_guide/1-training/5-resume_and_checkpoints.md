# Resume And Checkpoints

Checkpoint selection is controlled by algorithm-level fields. Use
`algo.load_run`, not `training.load_run`.

## Resume Training

PPO/APPO resume from `algo.load_run` directly. Use a run id or `-1` for the
latest run in the relevant log directory:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true
```

Off-policy algorithms (SAC/FlashSAC/WarpSAC) keep resume opt-in so that the
`algo.load_run=-1` default can never turn a fresh launch into an accidental
resume. Set `algo.resume=true` and select the checkpoint with `algo.load_run`
(run id or `-1` for the latest run) plus optional `algo.checkpoint`
(iteration or filename; `-1` selects the latest checkpoint):

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  --profile <owner-profile> \
  algo.resume=true \
  algo.load_run=2026-10-05_14-28-04_mjwarp \
  training.no_play=true
```

The off-policy checkpoint restores the full learner state (networks,
optimizers, schedulers, normalizers, and update count), and training
continues with absolute iteration numbering (`model_32000.pt` resumes at
iteration 32001, so `algo.max_iterations` stays the total budget, not an
increment). The replay buffer is not checkpointed: a resumed run re-warms
from an empty buffer through the usual train-start threshold, and RNG state
is not restored.

## Replay A Checkpoint

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` maps `--load-run` to the underlying checkpoint selector and sets
playback mode:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Some script paths accept a checkpoint path through `algo.load_run`; the unified
CLI validates `--load-run` as a run id and does not accept path separators.

## Seeds

Training seed resolution is implemented in `src/unilab/training/seed.py`.
Algorithm configs currently carry `algo.seed`, and the helper records seed
metadata when experiment tracking is active.
