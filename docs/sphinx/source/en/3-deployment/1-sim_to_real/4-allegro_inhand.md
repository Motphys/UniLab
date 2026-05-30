# Allegro / Sharpa In-Hand Manipulation Deployment

Cube reorientation on a 16-DoF Allegro hand or 17-DoF Sharpa hand. UniLab
trains these tactile-free — observations are joint state + cube pose only.

## What makes this hard

In-hand manipulation is the **friction-and-contact-sensitive** task. Tiny
deviations in:

- Cube edge geometry (corner radius, surface roughness)
- Finger pad friction (depends on temperature & humidity!)
- Joint backlash

can move a policy outside the regime it saw in simulation. Use the task-owned
domain-randomization config and validate the ranges in sim before hardware
bring-up; see {doc}`6-domain_randomization`.

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
Pose-estimation latency is deployment-stack specific. Measure it in the
hardware observation builder, then make the training owner and deploy runtime
agree on the observation timing before hardware deployment. See
{doc}`8-latency_budget`.
::::

## Grasp generator

Both `4-allegro_inhand` and `sharpa_inhand` envs ship a **grasp generator**
that samples plausible initial hand configurations. The hardware-side
equivalent is the operator placing the cube in the hand — verify your
distribution of starting configurations matches the trained env's grasp
generator output (see
`unilab.envs.manipulation.allegro_inhand.grasp_gen`).

If your real-world starting grip differs systematically, **add those poses
to the grasp generator**, retrain, and try again.

## Action interface

The manipulation envs map policy actions to joint position targets through the
task control config (`src/unilab/envs/manipulation/allegro_inhand/base.py` and
`src/unilab/envs/manipulation/sharpa_inhand/base.py`). The deploy controller
must use the same joint order, action scale, and limit policy.

## Failure recovery

A drop is unrecoverable without re-grasp. The hardware-side safety layer should
own the drop detector, using whatever cube-pose or force sensing the deploy
stack actually provides. On detection, enter the controller's safe state and
alert the operator.

## See also

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/8-manipulation/1-dexterous_inhand`
