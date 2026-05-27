# G1 Whole-Body Motion Tracking on Hardware

::::{admonition} Hardware target
:class: note
Unitree G1 humanoid (29-DoF variant). Joints assumed in the order the
URDF in `assets/robots/g1/` declares — verify with `unilab-export-scene`
before flashing.
::::

This guide walks the **last mile** between a converged G1 motion-tracking
policy and a closed-loop run on the robot.

## 0. Verify your sim-side checkpoint

```bash
# Replay the policy headlessly and produce a video.
uv run eval --algo ppo --task g1_motion_tracking --sim motrix \
    --load-run -1 --render-mode record --episodes 5
```

What to look for in the video:

- The pelvis stays inside a ±10 cm box around the reference trajectory.
- No "hand of god" recovery moves (large impulsive joint velocities).
- Foot strikes are roughly periodic — no chatter.

If any of those is off, fix it in sim. Hardware will *amplify* every defect.

## 1. Export

Pattern your script after `scripts/export_g1_motion_tracking.py` (see the
{doc}`onnx_export_and_runtime` page for the general procedure):

```bash
uv run python scripts/export_g1_motion_tracking.py \
    --run-dir runs/g1_motion_tracking_motrix/.../ \
    --out-dir deploy/g1/<checkpoint_name>/
```

The output directory should contain:

```
deploy/g1/<name>/
├── actor.onnx
├── obs_stats.npz
├── action_scale.npz
├── manifest.yaml
└── reference_motion.npz     # the motion clip the policy was trained on
```

## 2. Observation contract

The motion-tracking observation, in order, is:

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Group
  - Dim
  - Source on hardware
* - Joint position error (vs ref)
  - 29
  - encoder − reference clip phase-aligned to wall clock
* - Joint velocity
  - 29
  - encoder differentiated + low-pass (cutoff 50 Hz)
* - Projected gravity (base)
  - 3
  - IMU orientation → R\_world\_base · [0,0,-1]
* - Base angular velocity
  - 3
  - IMU gyro, bias-corrected
* - Phase variable
  - 2
  - sin / cos of motion clip time
* - Previous action
  - 29
  - last policy output (NOT measured joint pos!)
```

Hardware-side bring-up bug #1 is conflating *previous action* with *measured
joint position*. They differ by the position-error term in the PD law plus
unmodeled dynamics. Always feed back the **commanded** action.

## 3. Actuator interface

UniLab trains in **position-target** mode at the policy frequency (typically
50 Hz). The motor driver then runs a high-rate PD loop at ~2 kHz.

- Action = target joint position, **scaled** by `action_scale.npz`.
- Hard-clamp the target to the safe joint range on the **driver** side, not
  in the policy. The policy was trained against soft penalties; relying on
  Python-side clamping invites a delay glitch to push past the limit.

## 4. Reference motion sync

The phase variable lets the policy track an externally-supplied motion
clip. On hardware you need a wall-clock → phase mapping that is:

- **Monotonic** — no skipping back.
- **Restartable** — survives a comms hiccup without producing a step
  discontinuity in `(sin φ, cos φ)`.
- **Bounded rate** — clip dφ/dt to the value the policy was trained with
  (the motion loader records this; load `reference_motion.npz`).

See {py:mod}`unilab.envs.motion_tracking.g1.motion_loader` for the sim-side
loader you should mirror on hardware.

## 5. Safety layer

Hardware-side: see {doc}`safety_layers` for the standard structure. The G1
specifics:

- Reject actions whose Δ from previous > 0.3 rad — the policy never
  produces this, so it's almost certainly a comms corruption.
- Trip on |base roll| > 30° → soft-stop into a stable seated pose.
- Watchdog on policy heartbeat (must produce one action per 20 ms).

## 6. Closed-loop bring-up sequence

1. **Stand-on-stand**. Robot held by a gantry. Policy runs but actuators are
   torque-disabled. Confirm observation pipeline.
2. **Torque-enable, hand-held**. Operator catches the robot. Policy
   commands actuators. Confirm action mapping.
3. **Gantry-supported gait**. Track motion at half time-rate (dφ/dt halved).
4. **Free-stand**. Full rate, then remove gantry.

::::{admonition} Don't skip steps
:class: warning
We have seen 6/10 policies that pass step 1–3 fall in step 4 because of a
single inverted axis in the IMU mounting. Step 1 catches this.
::::

## 7. What to log

Log the **full observation vector**, **full action vector**, and **wall
clock** for every step. On a failure replay it against the sim model:

```bash
uv run python scripts/replay_hardware_trace.py \
    --trace deploy/g1/<name>/hw_run_2026-05-18.npz \
    --task g1_motion_tracking --sim motrix
```

A divergence point in joint position between sim-replay and the recorded
trace is the start of your bug.

## See also

- {doc}`onnx_export_and_runtime`
- {doc}`domain_randomization_for_real`
- {doc}`latency_and_observation_lag`
- {doc}`safety_layers`
- {doc}`../../user_guide/tasks/g1_motion_tracking`
