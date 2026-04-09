# G1 Motion Tracking

Languages: English | [简体中文](../zh_CN/05-g1-motion-tracking.md) | [日本語](../ja/05-g1-motion-tracking.md) | [한국어](../ko/05-g1-motion-tracking.md)

UniLab currently provides one whole-body motion-tracking task for G1.

- Hydra task: `g1_motion_tracking`
- Registered env name: `G1MotionTracking`
- Registered backends: `mujoco` and `motrix`
- Landed Motrix-specific configs: motion-tracking rewards for PPO and APPO
- Default motion file: `src/unilab/assets/motions/g1/dance1_subject2_part.npz`

## Environment Entrypoints

```bash
# PPO (RSL-RL, MuJoCo)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking

# PPO (RSL-RL, Motrix)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix

# APPO (MuJoCo)
uv run python scripts/train_appo.py task=g1_motion_tracking

# APPO (Motrix)
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix

# Play the latest checkpoint
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.play_only=true

# Motrix PPO playback opens the native renderer
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true

# APPO MuJoCo playback
uv run python scripts/train_appo.py task=g1_motion_tracking training.play_only=true

# APPO Motrix playback opens the native renderer
uv run python scripts/train_appo.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true
```

For G1 motion tracking, Motrix training and playback should primarily go through `scripts/train_rsl_rl.py` and `scripts/train_appo.py`. The debug script `scripts/play_interactive.py` still follows the MuJoCo viewer path.

## Interactive Debugging

`scripts/play_interactive.py` can visualize target bodies directly, and it can also show the reference pose and velocity used by the reward. The script is implemented against the MuJoCo viewer path and does not support the Motrix native renderer.

```bash
# Visualize motion targets
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_show_axes

# Inspect only selected bodies
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_body_names torso_link,left_wrist_yaw_link,right_wrist_yaw_link

# Show reward debug information
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_reward_debug \
  --reward_debug_show_velocity \
  --reward_debug_show_connectors \
  --target_max_bodies 4
```

If you need a specific run or checkpoint, also pass `--load_run` and `--checkpoint`.

## Motion Preprocessing

Training uses preprocessed `.npz` files. Use `scripts/motion/csv_to_npz.py` to convert Unitree-format CSV into an NPZ that the training environment can load:

```bash
# Full conversion
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50

# Export only a time slice
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

## Replay NPZ

After generating an NPZ, you can inspect it directly in the MuJoCo viewer with `scripts/motion/replay_npz.py`:

```bash
# Loop playback
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop

# Slow down to 0.5x
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

## Configuration Note

`task=g1_motion_tracking` reads the `motion_file` declared in the env config by default. To switch to a custom motion, first generate the `.npz`, then update the env config's default `motion_file`.

To validate a Motrix path, prefer the built-in play mode in the training scripts instead of the MuJoCo-only debug script:

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)
