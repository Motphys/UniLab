# Motion Tracking

G1 motion tracking tasks live under `src/unilab/tasks/motion_tracking/` and are
selected through task owner YAMLs in `conf/ppo/`, `conf/appo/`, and selected
off-policy paths.

> **Motion assets moved to Hugging Face.** The `.npz` clips are no longer shipped
> in the repository. On first use `MotionLoader`
> (`src/unilab/tasks/motion_tracking/common/motion_loader.py`) downloads them on demand
> from [unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)
> via `src/unilab/assets/hub.py` (`_HF_MOTIONS_REPO_ID`). `uv sync` already installs
> the required `huggingface_hub` dependency.

## Task Owners

Each task ships a default motion clip in its Hydra task-owner YAML. Hydra is the
configuration entry point; the selected owner is materialized into the shared
`ManagerBasedRlEnvCfg` and then consumed by the NumPy Manager-Based runtime.

| CLI Task | Registered Env | Default Motion | Owner Evidence |
| --- | --- | --- | --- |
| `g1_motion_tracking` | `G1MotionTracking` | `dance1_subject2_part.npz` | `conf/ppo/task/g1_motion_tracking/`, `conf/appo/task/g1_motion_tracking/` |
| `g1_flip_tracking` | `G1FlipTracking` | `flip_360_001__A304.npz` | `conf/ppo/task/g1_flip_tracking/`, `conf/appo/task/g1_flip_tracking/` |
| `g1_wall_flip_tracking` | `G1WallFlipTracking` | `flip_from_wall_104__A304.npz` | `conf/ppo/task/g1_wall_flip_tracking/`, `conf/appo/task/g1_wall_flip_tracking/` |
| `x2_wall_flip_tracking` | `X2WallFlipTracking` | `tictacflip_6-3_g1format.npz` | `conf/ppo/task/x2_wall_flip_tracking/` |
| `g1_climb_tracking` | `G1ClimbTracking` | `climb_20_z_scale_1.0.npz` | `conf/ppo/task/g1_climb_tracking/`, `conf/appo/task/g1_climb_tracking/` |
| `g1_box_tracking` | `G1BoxTracking` | `sub3_largebox_003_boxconverted.npz` | `conf/ppo/task/g1_box_tracking/` |
| `g1_wbt_obs` | `G1WBTObs` | `dance1_subject2_part.npz` | `conf/sac/task/g1_wbt_obs/mujoco.yaml` |

The 23-DoF task-owner directories select their matching 23-DoF scene, motion,
entity, and action declarations. Profile differences remain in Hydra. The G1
identities use the shared manager factory; X2 adds only a cold-path mesh resolver
before delegating to that factory.

## PPO And APPO

PPO owner iteration budgets (the `--sim mujoco` owner YAMLs): `g1_motion_tracking`
runs `algo.max_iterations=15000`; `g1_flip_tracking` and `g1_wall_flip_tracking`
run `20000`; `x2_wall_flip_tracking` runs `9500`. (The Motrix owner YAML for
`g1_flip_tracking` raises this to `30000`.)

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_wall_flip_tracking --sim mujoco
uv run train --algo ppo --task x2_wall_flip_tracking --sim mujoco
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

The `g1_wbt_obs` owner is the deploy-aligned off-policy observation profile. Its
actor keeps the command and anchor-orientation terms at one step while the
`base_ang_vel`, `joint_pos`, `joint_vel`, and `actions` terms declare
`history_length: 5`. `ObservationManager` owns and flattens those per-term
histories; the actor uses the configured encoder-biased joint-position term while
the critic keeps the clean term. Per-term oldest-first ordering is guarded by
`tests/scripts/test_obs_alignment_g1_wbt.py`; the hardware-side contract is
documented in the sim-to-real deployment guide. When a Motrix sim2sim replay needs
a checkpoint from another log root, pass the absolute path through `uv run eval`:

```bash
uv run eval --algo sac --task g1_motion_tracking --sim motrix \
  algo.load_run=/abs/path/to/logs/fast_sac/G1MotionTrackingSAC/2026-04-23_14-06-57_mujoco
```

## Motion Files

Motion NPZ files are selected through
`env.commands.motion.params.motion_file`, which accepts one path or a list of
paths. A standard clip must contain the seven keys `fps`, `joint_pos`,
`joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`
(validated in `common/motion_loader.py`):

```yaml
env:
  commands:
    motion:
      params:
        motion_file:
          - motions/g1/dance1_subject2_part.npz
          - motions/g1/walk1_subject5_from_csv.npz
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
  env.commands.motion.params.motion_file=motions/g1/motion_crawl_slope_uni.npz \
  env.scene.model_file=src/unilab/assets/robots/g1/scene_crawl_slope.xml \
  env.commands.motion.params.sampling_mode=start \
  env.commands.motion.params.truncate_on_clip_end=true \
  env.max_episode_seconds=20.0 \
  'env.commands.motion.params.pose_range={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  'env.commands.motion.params.velocity_range={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  'env.commands.motion.params.joint_position_range=[0,0]'
```

Key overrides: `env.commands.motion.params.motion_file` selects the crawl-slope
clip; `env.scene.model_file` switches to the slope scene
(`scene_crawl_slope.xml` exists under `src/unilab/assets/robots/g1/`);
`sampling_mode=start` plus `truncate_on_clip_end=true` starts from the clip
beginning and truncates there; and zeroing the command reset ranges reuses the
exact clip initial state.

## Interactive Debugging

Routine checkpoint replay uses `uv run eval`. When you need a target-body or reward
debug overlay, `scripts/play_interactive.py` is the low-level MuJoCo viewer entry
point; it is not currently exposed as a `uv run eval` flag.
