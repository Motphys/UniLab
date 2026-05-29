# G1 Whole-Body Motion Tracking on Hardware

::::{admonition} Hardware target
:class: note
Unitree G1 humanoid (29-DoF variant). Joints are assumed in the order exported
by `scripts/deploy/export_deploy_config.py` from
`src/unilab/assets/robots/g1/scene_flat.xml`; verify that order before
hardware bring-up.
::::

This guide walks the **last mile** between a converged G1 motion-tracking
policy and a closed-loop run on the robot.

## 0. Verify your sim-side checkpoint

```bash
# Replay the policy headlessly and produce a video.
uv run scripts/train_rsl_rl.py task=g1_motion_tracking/motrix \
  training.play_only=true \
  algo.load_run=-1
```

What to look for in the video:

- The tracked bodies follow the reference motion without large discontinuities.
- Joint velocities and actions remain finite and within the expected range.
- Contact timing looks consistent with the reference motion.

If any of those is off, fix the sim-side checkpoint or deploy contract before
hardware bring-up.

## 1. Export

Use the training playback path to export `policy.onnx`, then export the G1 WBT
deploy config and motion binary with the committed deployment helpers:

```bash
uv run scripts/train_rsl_rl.py task=g1_motion_tracking/motrix \
  training.play_only=true \
  algo.load_run=-1

uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/export_motion_bin.py \
  --output logs/deploy/dance1.bin
```

The deployment-side prototype consumes:

```
runs/<run>/
└── policy.onnx
logs/deploy/
├── deploy_config.yaml
└── dance1.bin
```

## 2. Observation contract

For the committed G1 WBT deploy helper, the observation layout is exported into
`deploy_config.yaml` as `obs_layout`. `scripts/deploy/export_deploy_config.py`
is the source of truth for the segment order:

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Group
  - Dim
  - Source on hardware
* - `command_joint_pos`
  - 29
  - motion reference frame joint position
* - `command_joint_vel`
  - 29
  - motion reference frame joint velocity
* - `motion_anchor_ori_b`
  - 6
  - anchor orientation term from the reference and robot torso frames
* - `gyro`
  - 3 per history step
  - IMU gyro term
* - `joint_pos_rel`
  - 29 per history step
  - measured joint position minus `default_angles`
* - `dof_vel`
  - 29 per history step
  - joint velocity term
* - `last_actions`
  - 29
  - previous raw actor output
```

The export script also records each segment's `history_length` and verifies the
total `obs_dim`. `scripts/deploy/sim_prototype.py` refuses to run when the ONNX
input width and `deploy_config.yaml` `obs_dim` disagree.

## 3. Actuator interface

The G1 deploy prototype maps actor output exactly as:
`action * action_scale + default_angles`, then clips to `joint_lower` /
`joint_upper` and applies EMA smoothing from `ema_alpha`.

- Action = target joint position, **scaled** by the `action_scale` entry in
  `deploy_config.yaml`.
- Clamp the target to the generated joint range before it reaches the motor
  driver.

## 4. Reference motion sync

The phase variable lets the policy track an externally-supplied motion
clip. On hardware you need a wall-clock → phase mapping that is:

- **Monotonic** — no skipping back.
- **Restartable** — survives a comms hiccup without producing a step
  discontinuity in `(sin φ, cos φ)`.
- **Bounded rate** — clip dφ/dt to the value the policy was trained with
  (the motion loader records this; load `reference_motion.npz`).

See `unilab.envs.motion_tracking.g1.motion_loader` for the sim-side
loader you should mirror on hardware.

## 5. Safety layer

Hardware-side: see {doc}`safety_layers` for the standard structure. The G1
specifics:

- Reject non-finite actions and shape mismatches before applying
  `action_scale`.
- Clamp generated targets with `joint_lower` / `joint_upper` from
  `deploy_config.yaml`.
- Keep watchdog, pose monitor, and operator-stop thresholds in the deploy
  controller and test them independently of the policy.

## 6. Closed-loop bring-up sequence

1. **Stand-on-stand**. Robot held by a gantry. Policy runs but actuators are
   torque-disabled. Confirm observation pipeline.
2. **Torque-enable, hand-held**. Operator catches the robot. Policy
   commands actuators. Confirm action mapping.
3. **Gantry-supported gait**. Track motion at half time-rate (dφ/dt halved).
4. **Free-stand**. Full rate, then remove gantry.

Do not skip the observation-only stage: it is where axis-order, joint-order,
and `last_actions` wiring mistakes are easiest to catch.

## 7. What to log

Log the **full observation vector**, **full action vector**, and **wall
clock** for every step. Before hardware bring-up, validate the same ONNX,
deploy config, and motion binary through the MuJoCo deployment prototype:

```bash
uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

A mismatch between the ONNX input width and `deploy_config.yaml` `obs_dim` is a
deployment contract bug, not a hardware tuning problem.

## See also

- {doc}`onnx_runtime`
- {doc}`domain_randomization`
- {doc}`latency_budget`
- {doc}`safety_layers`
- {doc}`../../user_guide/tasks/motion_tracking`
