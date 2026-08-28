# G1 Whole-Body Motion Tracking on Hardware

::::{admonition} Hardware target
:class: note
Unitree G1 humanoid (29-DoF variant). Joint order comes from the task owner's
scene (`src/unilab/assets/robots/g1/scene_flat.xml`, actuator order); verify
that order against your SDK motor indices before hardware bring-up.
::::

This guide covers the **observation and action contract** a G1 motion-tracking
policy expects on hardware. The repository does not ship a G1 deploy runtime —
you supply the hardware-side loop, and this page tells you what it must
reproduce.

## 0. Verify your sim-side checkpoint

```bash
# Replay the policy headlessly and produce a video.
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1 \
  --render-mode record
```

What to look for in the video:

- The tracked bodies follow the reference motion without large discontinuities.
- Joint velocities and actions remain finite and within the expected range.
- Contact timing looks consistent with the reference motion.

If any of those is off, fix the sim-side checkpoint before hardware bring-up.

## 1. Export, then read the contract off the owner YAML

Export `policy.onnx` through the training playback path, using the same task
owner that produced the checkpoint:

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1
```

Every field your hardware loop needs is declared in that owner's YAML. Widths
differ per owner — two G1 examples:

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - Owner
  - Actor obs width
  - Notes
* - `conf/sac/task/g1_wbt_obs/mujoco.yaml`
  - 514
  - No state estimation: `motion_anchor_pos_b` and `base_lin_vel` are set to
    `null`, pelvis IMU, `history_length: 5` on the proprio terms.
* - `conf/ppo/task/g1_motion_tracking_deploy/mujoco.yaml`
  - 154
  - Single-step mimic actor layout, per-joint-group `scale` regex map.
```

::::{admonition} Read the width off the composed config, not off this table
:class: warning
Actor obs width is the sum of `dim * history_length` over the terms declared
under `env.observations.actor.terms`, plus the motion command. A term set to
`null` is dropped. If the ONNX input width disagrees with what your hardware
loop assembles, that is a contract bug, not a hardware tuning problem.
::::

## 2. Observation contract

Term order and per-term history come from your own owner's
`env.observations.actor.terms`. As a worked example, `g1_wbt_obs` declares the
terms below; those carrying `history_length: 5` are flattened **oldest-first**
within the term, and terms are concatenated in declaration order:

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Term
  - Dim
  - Source on hardware
* - `motion_anchor_ori_b`
  - 6
  - anchor orientation term from the reference and robot torso frames
* - `base_ang_vel`
  - 3 per history step
  - IMU gyro (`params.sensor_name: pelvis_gyro`)
* - `joint_pos`
  - 29 per history step
  - measured joint position minus the `stand` keyframe joint angles
* - `joint_vel`
  - 29 per history step
  - joint velocity term
* - `actions`
  - 29 per history step
  - previous raw actor output
```

The motion command contributes the reference joint position and velocity
(`29 + 29`) ahead of the observation terms. Per-term oldest-first ordering is
guarded by `tests/scripts/test_obs_alignment_g1_wbt.py`; mirror that ordering
on hardware or the policy reads a permuted vector.

## 3. Actuator interface

Map actor output as `action * scale + default_angles`, then clamp to the
scene's joint range before the target reaches the motor driver.

- `scale` is `env.actions.joint_pos.scale`. It may be a **scalar** (`2.0` for
  `g1_wbt_obs`) or a **regex → value map** resolved per actuator
  (`g1_motion_tracking_deploy` maps joint-name patterns to distinct values).
  Reproduce the owner's resolved per-actuator vector exactly — do not average a
  map, take one entry, or broadcast a scalar over a map owner.
- `default_angles` follows from `use_default_offset: true`, i.e. the `stand`
  keyframe joint block of the owner's scene.
- Joint limits and gains come from the same scene XML (`jnt_range`, position
  actuator `gainprm` / `biasprm`).

Training applies the target directly with no smoothing. If hardware jitter
forces you to add smoothing, verify the sim2sim impact first — every step of lag
pushes observations out of the training distribution.

## 4. Reference motion sync

The phase variable lets the policy track an externally-supplied motion
clip. On hardware you need a wall-clock → phase mapping that is:

- **Monotonic** — no skipping back.
- **Restartable** — survives a comms hiccup without producing a step
  discontinuity in `(sin φ, cos φ)`.
- **Bounded rate** — clip dφ/dt to the value the policy was trained with
  (the motion loader records this; load `reference_motion.npz`).

See `unilab.tasks.motion_tracking.common.motion_loader` for the sim-side
loader you should mirror on hardware.

## 5. Safety layer

Hardware-side: see {doc}`7-safety_layers` for the standard structure. The G1
specifics:

- Reject non-finite actions and shape mismatches before applying the action
  scale.
- Clamp generated targets with the joint range from the owner's scene XML.
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
and `actions` wiring mistakes are easiest to catch.

## 7. What to log

Log the **full observation vector**, **full action vector**, and **wall
clock** for every step. Compare the first hardware observation window against a
sim episode built from the same owner YAML — that diff localizes unit, frame,
and ordering mistakes faster than any reward inspection.

## See also

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`8-latency_budget`
- {doc}`7-safety_layers`
- {doc}`../../2-user_guide/4-tasks/2-motion_tracking`
