# Motion Tracking

G1 motion tracking tasks live under `src/unilab/envs/motion_tracking/` and are
selected through task owner YAMLs in `conf/ppo/`, `conf/appo/`, and selected
off-policy paths.

> **Motion assets moved to Hugging Face.** The `.npz` clips are no longer shipped
> in the repository. On first use `MotionLoader`
> (`src/unilab/envs/motion_tracking/g1/motion_loader.py`) downloads them on demand
> from [unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)
> via `src/unilab/assets/hub.py` (`_HF_MOTIONS_REPO_ID`). `uv sync` already installs
> the required `huggingface_hub` dependency.

## Task Owners

Each task ships a default motion clip defined in the env config dataclass:

| CLI Task | Registered Env | Default Motion | Owner Evidence |
| --- | --- | --- | --- |
| `g1_motion_tracking` | `G1MotionTracking` | `dance1_subject2_part.npz` | `conf/ppo/task/g1_motion_tracking/`, `conf/appo/task/g1_motion_tracking/` |
| `g1_flip_tracking` | `G1FlipTracking` | `flip_360_001__A304.npz` | `conf/ppo/task/g1_flip_tracking/`, `conf/appo/task/g1_flip_tracking/` |
| `g1_wall_flip_tracking` | `G1WallFlipTracking` | `flip_from_wall_104__A304.npz` | `conf/ppo/task/g1_wall_flip_tracking/`, `conf/appo/task/g1_wall_flip_tracking/` |
| `g1_climb_tracking` | G1 climb tracking env | clip from env config | `conf/ppo/task/g1_climb_tracking/`, `conf/appo/task/g1_climb_tracking/` |
| `g1_box_tracking` | G1 box tracking env | clip from env config | `conf/ppo/task/g1_box_tracking/` |
| `g1_wbt_obs` | `G1MotionTrackingSAC` | shared with `g1_motion_tracking` | `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` |

The defaults are set in code: `dance1_subject2_part.npz`
(`g1/tracking.py`), `flip_360_001__A304.npz` and `flip_from_wall_104__A304.npz`
(`g1/flip_tracking.py`).

## PPO And APPO

PPO owner iteration budgets (the `--sim mujoco` owner YAMLs): `g1_motion_tracking`
runs `algo.max_iterations=15000`; `g1_flip_tracking` and `g1_wall_flip_tracking`
run `20000`. (The Motrix owner YAML for `g1_flip_tracking` raises this to `30000`.)

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_wall_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  algo.num_envs=128 algo.max_iterations=5 training.no_play=true
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1 \
  training.cam_tracking=true training.cam_tracking_env_idx=0
```

## SAC WBT Path

```bash
uv run train --algo sac --task g1_motion_tracking --sim mujoco training.use_amp=true
uv run train --algo sac --task g1_wbt_obs --sim mujoco training.use_amp=true
```

The `g1_wbt_obs` owner is the deploy-aligned off-policy observation profile: a
pelvis IMU state (`pelvis_local_linvel` / `pelvis_gyro` / `pelvis_upvector`) plus
per-term observation history (`noise_config.obs_history_length: 5`), byte-aligned
with the deploy-time `ObservationManager`. Deploy tooling lives under
`scripts/deploy/`, and the observation alignment is cross-checked by
`tests/scripts/test_obs_alignment_g1_wbt.py`. When a Motrix sim2sim replay needs a
checkpoint from another log root, pass the absolute path through `uv run eval`:

```bash
uv run eval --algo sac --task g1_motion_tracking --sim motrix \
  algo.load_run=/abs/path/to/logs/fast_sac/G1MotionTrackingSAC/2026-04-23_14-06-57_mujoco
```

## Motion Files

Motion NPZ files are read through `env.motion_file`, which also accepts a list of
paths. A standard clip must contain the seven keys `fps`, `joint_pos`,
`joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`
(validated in `g1/motion_loader.py`):

```yaml
env:
  motion_file:
    - src/unilab/assets/motions/g1/dance1_subject2_part.npz
    - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

Conversion and inspection helpers are in `scripts/motion/`:

```bash
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 --output_fps 50
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 --output_fps 50 --start_time 4.0 --end_time 9.0
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --loop
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --speed 0.5
```

If a MuJoCo replay shows obviously displaced bodies, check first: whether the NPZ
holds all seven keys, whether `fps` matches the control frequency, whether the body
layout needs a remap, and whether the joint order matches the current G1 model.
For more detailed motion conversion notes, see `scripts/motion/README.md`.

## SAC WBT On Crawl-Slope Scene

Running `g1_motion_tracking` on slope terrain requires switching both the motion
clip and the MuJoCo scene file, fixing the episode length, and disabling reset
randomization so the precise clip start state is reused:

```bash
CUDA_VISIBLE_DEVICES=1 uv run train --algo sac --task g1_motion_tracking --sim mujoco \
  training.use_amp=true algo.seed=1 \
  +env.motion_file=src/unilab/assets/motions/g1/motion_crawl_slope_uni.npz \
  +env.scene.model_file=src/unilab/assets/robots/g1/scene_crawl_slope.xml \
  +env.sampling_mode=start \
  env.truncate_on_clip_end=true \
  +env.max_episode_seconds=20.0 \
  '+env.pose_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.velocity_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.joint_position_range=[0,0]'
```

Key overrides: `env.motion_file` selects the crawl-slope clip;
`env.scene.model_file` switches to the slope scene (`scene_crawl_slope.xml` exists
under `src/unilab/assets/robots/g1/`); `sampling_mode=start` plus
`truncate_on_clip_end=true` starts from the clip beginning and truncates there; and
zeroing the randomization ranges reuses the exact clip initial state.

## Interactive Debugging

Routine checkpoint replay uses `uv run eval`. When you need a target-body or reward
debug overlay, `scripts/play_interactive.py` is the MuJoCo-only low-level entry
point; it is not currently exposed as a `uv run eval` flag.
