# Allegro / Sharpa In-Hand Manipulation Deployment

Cube reorientation on a 16-DoF Allegro hand or 17-DoF Sharpa hand. UniLab
trains these tactile-free — observations are joint state + cube pose only.

## What makes this hard

In-hand manipulation is the **friction-and-contact-sensitive** task. Tiny
deviations in:

- Cube edge geometry (corner radius, surface roughness)
- Finger pad friction (depends on temperature & humidity!)
- Joint backlash

…break a policy that worked perfectly in sim. Aggressive DR is mandatory:
see {doc}`domain_randomization_for_real` for the recipe.

## Observation contract

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Group
  - Dim
  - Source on hardware
* - Joint positions
  - 16 (Allegro) / 17 (Sharpa)
  - encoder
* - Joint velocities
  - 16 / 17
  - encoder differentiated, low-pass
* - Cube pose (world)
  - 7
  - vision (RGB-D + pose estimator, or fiducial)
* - Cube linear/angular velocity
  - 6
  - finite-difference of pose, low-pass; **noisy on real**
* - Target rotation quaternion
  - 4
  - command
* - Previous action
  - 16 / 17
  - last policy output
```

::::{admonition} Vision pipeline latency
:class: warning
Pose estimators introduce 30–80 ms of delay. The policy was trained with
small or zero delay by default — re-train with realistic delay injection
*before* hardware deployment. See {doc}`latency_and_observation_lag`.
::::

## Grasp generator

Both `allegro_inhand` and `sharpa_inhand` envs ship a **grasp generator**
that samples plausible initial hand configurations. The hardware-side
equivalent is the operator placing the cube in the hand — verify your
distribution of starting configurations matches the trained env's grasp
generator output (see
{py:mod}`unilab.envs.manipulation.allegro_inhand.grasp_gen`).

If your real-world starting grip differs systematically, **add those poses
to the grasp generator**, retrain, and try again.

## Action interface

Position targets at ~30 Hz, low-rate PD on the hand controller. The hand
firmware imposes its own torque limits; ensure they envelope the policy's
output range.

## Failure recovery

A drop is unrecoverable without re-grasp. Hardware-side safety layer
detects drop via:

- Wrist force-torque magnitude < `cube_mass * g * 0.5` for >100 ms, OR
- Visual cube pose drops below palm height.

On detection: zero torque, alert operator.

## See also

- {doc}`onnx_export_and_runtime`
- {doc}`domain_randomization_for_real`
- {doc}`../../user_guide/manipulation/dexterous_inhand`
