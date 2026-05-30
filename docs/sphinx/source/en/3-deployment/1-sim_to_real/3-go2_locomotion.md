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
  - encoder velocity after the deploy controller's filtering path
* - Previous action
  - 12 / 16
  - last policy output
* - Foot contact
  - 4 (Go2 only)
  - contact sensor or estimated from foot height
```

::::{admonition} State estimator caveat
:class: warning
The policy is trained against the observation terms emitted by the selected env
owner. If deployment cannot provide the same base-velocity signal, train a
variant whose actor observation matches the estimator you can run on the robot
(see HIM-PPO at {doc}`../../2-user_guide/2-algorithms/6-him_ppo`).
::::

## Rough terrain caveat

For `go2_joystick_rough` the policy expects elevated terrain features. On a
flat indoor surface the rough-trained policy will be *more conservative*
than necessary but should still be validated through replay before hardware
bring-up. For deployment on slopes / debris:

- Choose ground-friction DR ranges from measured deployment surfaces.
- Train with terrain curriculum: see
  {doc}`../../2-user_guide/6-terrain/1-procedural`.

## Go2W wheel ↔ leg dispatch

Go2W policies output **continuous wheel velocity** for the rear wheel
joints and **position targets** for the legs. The action vector ordering
must match `src/unilab/assets/robots/go2w/`. Verify with `unilab-export-scene`.

## See also

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/4-tasks/1-locomotion`
