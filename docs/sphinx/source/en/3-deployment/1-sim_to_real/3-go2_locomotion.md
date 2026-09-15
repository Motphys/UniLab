# Go2 Locomotion Deployment

This page describes the retained `go2_joystick_flat` reference owner. Unitree
production deployment variants are documented by their ecosystem package.

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
  - 12
  - encoder
* - Joint velocities
  - 12
  - encoder velocity after the deploy controller's filtering path
* - Previous action
  - 12
  - last policy output
* - Foot contact
  - 4
  - contact sensor or estimated from foot height
```

::::{admonition} State estimator caveat
:class: warning
The policy is trained against the observation terms emitted by the selected env
owner. If deployment cannot provide the same base-velocity signal, train a
variant whose actor observation matches the estimator you can run on the robot.
::::

## See also

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/4-tasks/1-locomotion`
