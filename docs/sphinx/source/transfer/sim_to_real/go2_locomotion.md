# Go2 / Go2W Locomotion Deployment

Joystick-driven locomotion (flat + rough) plus the wheel-legged Go2W
variant. The hardware story for both is similar; this page calls out the
deltas.

## Observation contract

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Group
  - Dim
  - Source on hardware
* - Base linear velocity
  - 3
  - state estimator (KF over IMU + leg odometry); NOT raw integration
* - Base angular velocity
  - 3
  - IMU gyro
* - Projected gravity
  - 3
  - IMU orientation
* - Joystick command (vx, vy, ωz)
  - 3
  - operator input
* - Joint positions
  - 12 (Go2) / 16 (Go2W)
  - encoder
* - Joint velocities
  - 12 / 16
  - encoder, low-pass 50 Hz
* - Previous action
  - 12 / 16
  - last policy output
* - Foot contact
  - 4 (Go2 only)
  - contact sensor or estimated from foot height
```

::::{admonition} State estimator caveat
:class: warning
The policy is *trained against the simulator's ground-truth base velocity*.
A poor real-world state estimator is the #1 cause of jittery walking. If
you do not have a working KF, train a separate variant of the policy with
**estimated** velocity in observation (see HIM-PPO at
{doc}`../../user_guide/algorithms/him_ppo`).
::::

## Rough terrain caveat

For `go2_joystick_rough` the policy expects elevated terrain features. On a
flat indoor surface the rough-trained policy will be *more conservative*
than necessary but will still run. For deployment on slopes / debris:

- DR ranges for ground friction should include μ = 0.3 (wet tile).
- Train with terrain curriculum: see
  {doc}`../../user_guide/terrain/procedural`.

## Go2W wheel ↔ leg dispatch

Go2W policies output **continuous wheel velocity** for the rear wheel
joints and **position targets** for the legs. The action vector ordering
must match `assets/robots/go2w/`. Verify with `unilab-export-scene`.

## See also

- {doc}`onnx_export_and_runtime`
- {doc}`domain_randomization_for_real`
- {doc}`../../user_guide/tasks/locomotion_zoo`
