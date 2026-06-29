# OpenArm Pick

This page covers the checked-in OpenArm single-arm pick task: how to train, how
to evaluate, and how to record playback video. Select backends with `--task` and
`--sim`; do not override `training.sim_backend` alone. The owner YAMLs remain the
internal evidence for which combinations are configured.

The task uses the registered env `OpenArmDemoPick`
(`@registry.env("OpenArmDemoPick", sim_backend="mujoco")`, see
`src/unilab/envs/manipulation/openarm/pick_place_demo.py`). Only the MuJoCo owner
path is checked in; there is no Motrix owner. The left arm is the only movable
arm (`left_arm_only`, `rigid_freeze_right_arm`, `fix_lifter`); the policy controls
the 7 left-arm joints plus the gripper.

## Owner configs

The config group is `openarm_demo_pick`; all owner YAMLs live under
`conf/ppo/task/openarm_demo_pick/`:

| File | `--profile` | Notes |
| --- | --- | --- |
| `mujoco.yaml` | (none) | Base pick task inherited by the variants; not meant to train on its own |
| `mujoco_lift3d.yaml` | `lift3d` | Main task: 3 cm cube, in-air 3D goal, binary gripper, proximity-gated dense lift |
| `mujoco_lift3d_easy.yaml` | `lift3d_easy` | Curriculum/debug aid: lower, wider goal (`goal z 1.13→1.09`, `success_dist 0.05→0.06`) |
| `mujoco_lift3d_contgrip.yaml` | `lift3d_contgrip` | Continuous-gripper variant: policy closes the gripper continuously, with staged + firm grasp shaping |
| `mujoco_lift3d_contgrip_lowent.yaml` | `lift3d_contgrip_lowent` | Same task/reward as contgrip, only lowers `entropy_coef` 0.01→0.003: removes the late-training `action std`/`entropy` drift with equal/slightly-better success (see the control run notes in the file) |

The top-level CLI selects a variant with `--profile`: `--task openarm_demo_pick
--sim mujoco --profile lift3d_contgrip` composes the owner
`openarm_demo_pick/mujoco_lift3d_contgrip`.

## Training

Train the continuous-gripper main variant (no rendering):

```bash
uv run train --algo ppo --task openarm_demo_pick --sim mujoco \
  --profile lift3d_contgrip training.no_play=true
```

The binary-gripper lift3d main task and the easier curriculum variant:

```bash
uv run train --algo ppo --task openarm_demo_pick --sim mujoco \
  --profile lift3d training.no_play=true
uv run train --algo ppo --task openarm_demo_pick --sim mujoco \
  --profile lift3d_easy training.no_play=true
```

The owner defaults to `max_iterations: 1500` and `num_envs: 4096`. Runs land in
`logs/rsl_rl_ppo/OpenArmDemoPick/<timestamp>_mujoco/`.

## Recording playback video

The `eval` route enters playback with `play_only=true`; on a headless machine the
default `play_render_mode` resolves to recording, writing `play_video.mp4` into
the run directory. `--load-run -1` picks the latest run:

```bash
uv run eval --algo ppo --task openarm_demo_pick --sim mujoco \
  --profile lift3d_contgrip --load-run -1
```

The owner YAML already sets playback parameters for this task: `play_video_fps: 5`
(~10x slow motion over 50 Hz physics, easier to inspect the grasp),
`play_hide_geom_groups: [3]` (hide the cell enclosure mesh so it does not occlude
the arm), `play_steps: 200`, and the camera pose. Override on the command line
with `training.play_steps=300` etc.

## Evaluation metrics

The top-level `eval` only plays back the single on-camera env in the video. For
objective pick metrics across many parallel envs, use
`scripts/eval_openarm_success.py`: it loads a checkpoint, rolls the deterministic
policy out over many envs (no rendering), and reports ever-success, final hold
success, drop rate, final cube height, grasp firmness, and staged-grasp
adherence. `eval_envs` is not in the schema, so append it with `+`:

```bash
HIP_VISIBLE_DEVICES=0 uv run scripts/eval_openarm_success.py \
  task=openarm_demo_pick/mujoco_lift3d_contgrip \
  algo.load_run=logs/rsl_rl_ppo/OpenArmDemoPick/<run>_mujoco \
  +training.eval_envs=512 training.play_steps=200
```

## Helper scripts

- `scripts/openarm_scripted_pick.py`: a non-RL scripted staged pick (APPROACH →
  DESCEND → CLOSE → LIFT) using a damped least-squares IK delta on the left-arm
  Jacobian (`openarm_left_tcp` site). It runs through the same
  `run_playback_mode` pipeline as the policy and needs no checkpoint, producing a
  clean reference demo video.

  ```bash
  MUJOCO_GL=egl HIP_VISIBLE_DEVICES=0 uv run scripts/openarm_scripted_pick.py \
    task=openarm_demo_pick/mujoco_lift3d_contgrip training.play_steps=260
  ```

- `scripts/verify_openarm_play_motion.py`: checks that the left-arm joints
  actually move in a physics rollout (env 0), a regression guard against the
  "policy never moves the arm" degeneration.

  ```bash
  HIP_VISIBLE_DEVICES=0 uv run scripts/verify_openarm_play_motion.py \
    --run-dir logs/rsl_rl_ppo/OpenArmDemoPick/<run>_mujoco --steps 120
  ```

## Training results and curve comparison

`lift3d_contgrip_lowent` (`entropy_coef=0.003`) vs the baseline
`lift3d_contgrip` (`entropy_coef=0.01`) (PPO, seed=1, 4096 env x 24 steps x 1500
iter ~= 147.5M steps):

| Metric | baseline 0.01 | lowent 0.003 |
| --- | --- | --- |
| ever success (512-env deterministic eval) | 98.8% | 100.0% |
| final success | 86.3% | 87.9% |
| drop rate | 0% | 0% |
| final reward | 2580 | 2800 |
| final `action std` | 39.08 | 1.35 |
| final `entropy loss` | ~40 (monotonic climb) | ~12 (flat) |

Lowering `entropy_coef` from 0.01 to 0.003 removes the `action std` / `entropy`
drift that PPO accrues after the task is solved (iter ~600): tanh saturation lets
the policy inflate the pre-squash std for "free" entropy bonus without changing
the executed action. The result is a cleaner training curve with equal/slightly
better deterministic success (eval uses the policy mean, so it is unaffected by
exploration noise). The only cost is reaching the plateau ~150 iter later.

![entropy_coef 0.01 vs 0.003 training curves](../../../_static/images/openarm_pick_lowent_curves.png)

The learned grasp is a stable open fingertip-cradle (closure ~0); for this
high-friction small-cube geometry the open cradle is the more robust optimum:

![gripper cradling the cube, left-arm right-front close-up](../../../_static/images/openarm_pick_grasp_closeup.png)

For the category-level task page, see {doc}`../4-tasks/3-manipulation`.
