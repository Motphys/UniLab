# Motion Tracking

G1 motion tracking tasks live under `src/unilab/envs/motion_tracking/` and are
selected through task owner YAMLs in `conf/ppo/`, `conf/appo/`, and selected
off-policy paths.

## Task Owners

| CLI Task | Registered Env | Owner Evidence |
| --- | --- | --- |
| `g1_motion_tracking` | `G1MotionTracking` | `conf/ppo/task/g1_motion_tracking/`, `conf/appo/task/g1_motion_tracking/` |
| `g1_flip_tracking` | `G1FlipTracking` | `conf/ppo/task/g1_flip_tracking/`, `conf/appo/task/g1_flip_tracking/` |
| `g1_wall_flip_tracking` | `G1WallFlipTracking` | `conf/ppo/task/g1_wall_flip_tracking/`, `conf/appo/task/g1_wall_flip_tracking/` |
| `g1_climb_tracking` | G1 climb tracking env | `conf/ppo/task/g1_climb_tracking/`, `conf/appo/task/g1_climb_tracking/` |
| `g1_box_tracking` | G1 box tracking env | `conf/ppo/task/g1_box_tracking/` |
| `g1_wbt_obs` | `G1MotionTrackingSAC` | `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` |

## PPO And APPO

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_motion_tracking --sim motrix training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
```

## SAC WBT Path

```bash
uv run train --algo sac --task g1_wbt_obs --sim mujoco training.use_amp=true
uv run train --algo sac --task g1_wbt_obs --sim mujoco \
  training.use_amp=true
```

The `g1_wbt_obs` owner is the deploy-aligned off-policy observation profile.

## Motion Files

Motion NPZ files are read through `env.motion_file`. The expected training
payload includes `fps`, joint position/velocity, body pose, and body velocity
arrays. Conversion and inspection helpers are in `scripts/motion/`:

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop
```

For more detailed motion conversion notes, see `scripts/motion/README.md`.
