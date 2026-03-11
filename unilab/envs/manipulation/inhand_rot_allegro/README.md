# AllegroInhandRotation

Dexterous in-hand rotation of a tennis ball with the Allegro right hand in MuJoCo.
The policy learns to spin the ball continuously around a target axis (default: +Z, i.e. counterclockwise when viewed from above) using 16-DOF position control.

---

## Overview

| Item | Detail |
|---|---|
| Robot | Allegro Hand (right), 16 DOF |
| Object | YCB 056 Tennis Ball (sphere, r = 4 cm, m = 50 g) |
| Control | Incremental position targets, 20 Hz (ctrl\_dt = 0.05 s) |
| Physics | MuJoCo, sim\_dt = 5 ms (10 substeps per control step) |
| Episode length | 20 s |
| Observation | 3-frame lag history of [joint\_pos\_norm(16), targets(16), ball\_pos(3)] = 105-dim |
| Action | 16-dim delta joint targets ∈ [−1, 1], scaled by 1/24 rad |

---

## Quick Start

### Step 1 — Generate grasp poses

The RL policy resets from a cache of pre-collected stable grasp states.
A pre-generated cache (`grasp_50k.npy`, 50 000 grasps) is included.
To regenerate it (e.g. after changing the hand model or keyframe):

```bash
# From the UniLab root directory
python unilab/envs/manipulation/inhand_rot_allegro/gen_grasp.py
```

**Key options:**

| Argument | Default | Description |
|---|---|---|
| `--num_envs` | 2048 | Parallel MuJoCo environments |
| `--target` | 50000 | Number of grasps to collect |
| `--joint_noise` | 0.25 | ±rad noise at each reset for diverse exploration |
| `--quality_check` | on | Filter by fingertip-ball distance (pass `--no_quality_check` to disable) |
| `--output` | `grasps/grasp_50k.npy` | Output path |
| `--viewer` | off | Open live MuJoCo viewer (use with small `--num_envs`) |
| `--record_video` | off | Save a preview video of the collection |

The output is a `(N, 23)` float32 array:
`[hand_qpos(16), ball_pos(3), ball_quat(4)]`

---

### Step 2 — Train RL

```bash
# From the UniLab root directory
python scripts/train_rsl_rl.py --task AllegroInhandRotation
```

**Common options:**

```bash
# Override number of environments
python scripts/train_rsl_rl.py --task AllegroInhandRotation --env_num 8192

# Resume from a previous run
python scripts/train_rsl_rl.py --task AllegroInhandRotation \
    --load_run 2026-03-10_22-50-13

# Train without rendering a play video afterwards
python scripts/train_rsl_rl.py --task AllegroInhandRotation --no_play
```

Training logs are saved to `logs/rsl_rl_train/AllegroInhandRotation/<timestamp>/`.

---

### Step 3 — Evaluate / render video

```bash
python scripts/train_rsl_rl.py --task AllegroInhandRotation --play_only

# Custom camera
python scripts/train_rsl_rl.py --task AllegroInhandRotation --play_only \
    --cam_distance 1.5 --cam_elevation -30 --cam_azimuth 45

# Load a specific run
python scripts/train_rsl_rl.py --task AllegroInhandRotation --play_only \
    --load_run 2026-03-10_22-50-13
```

The video is saved as `play_video.mp4` inside the loaded run directory.

---

## Reward

| Term | Scale | Description |
|---|---|---|
| `rotate` | +1.25 | Ball angular velocity along rotation axis, clipped to ±0.5 rad/s |
| `obj_linvel` | −0.3 | L1 penalty on ball linear velocity (encourages pure spin) |
| `pose_diff` | −0.3 | L2 penalty on hand joint drift from initial grasp |
| `torque` | −0.1 | L2 penalty on PD torques |
| `work` | −2.0 | Penalty on mechanical work (torque · velocity) |

All terms are multiplied by `ctrl_dt` (0.05 s) before accumulation.

---

## File Structure

```
inhand_rot_allegro/
├── base.py            # AllegroBaseMjEnv: PD control, state accessors
├── rotation.py        # AllegroRotationMj: rewards, observations, reset
├── gen_grasp.py       # Grasp cache generation script
├── grasps/
│   └── grasp_50k.npy  # Pre-generated grasp cache (50 000 states)
└── xml/
    ├── scene.xml       # Top-level scene (includes hand + ball)
    ├── allegro_right.xml  # Hand model, defaults, actuators
    └── ball.xml        # Tennis ball body
```

---

## Configuration

Key parameters are in `rotation.py` (`RewardConfig`, `DomainRandConfig`) and `unilab/config/manipulation_params.py` (`rsl_rl_config`).

**PPO hyperparameters** (in `manipulation_params.py`):

| Parameter | Value |
|---|---|
| `max_iterations` | 10 000 |
| `num_steps_per_env` | 8 |
| `learning_rate` | 1e-3 (adaptive) |
| `entropy_coef` | 0.01 |
| `desired_kl` | 0.02 |
| `empirical_normalization` | True |
