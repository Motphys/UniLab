# Hardware Safety Layers

The policy produces actions under the training contract. A deploy-side safety
layer must live **between the policy output and the motor driver** and reject
contract violations before they become actuator commands.

## Required components

```{list-table}
:header-rows: 1
:widths: 30 70

* - Layer
  - Responsibility
* - Schema check
  - Action has correct dtype, shape, finite values. Reject NaN / Inf.
* - Range clamp
  - Clamp each joint target to deploy-configured joint limits.
* - Δ clamp
  - Reject or clamp per-step action deltas using a threshold owned by the
    deploy controller.
* - Rate limit
  - Slew-rate limit applied AFTER clamp.
* - Watchdog
  - If no fresh action arrives within the controller-owned timeout, hold the
    last known safe target or enter the controller's safe state.
* - Pose monitor
  - Roll / pitch outside operating envelope → triggered fault.
* - Operator stop
  - Big red button → instant torque-disable, regardless of state.
```

## Where the safety layer lives

```{mermaid}
flowchart LR
    P[Policy ONNX] --> S[Safety layer<br/>C++ on robot computer]
    S -->|safe target| D[Motor driver]
    D -->|encoder + IMU| Pre[Observation builder]
    Pre --> P
    S -.->|fault| OP[Operator UI]
    OP -.->|E-stop| D
```

Keep the hard real-time safety checks in the deploy controller, not in the
training script. The repository does not implement a production motor-driver
safety loop — that boundary is yours to build and test.

## What the policy assumes you've configured

The policy expects the action mapping and limits its training owner declared.
For the G1 WBT owner (`src/unilab/conf/sac/task/g1_wbt_obs/mujoco.yaml`):

| Quantity | Authority |
| --- | --- |
| action scale | `env.actions.joint_pos.scale` (scalar `2.0` for this owner; other owners declare a regex → value map resolved per actuator) |
| default angles | `use_default_offset: true`, i.e. the `stand` keyframe joint block of the owner's scene XML |
| joint limits | `jnt_range` in the scene XML |
| `kp` / `kd` | position actuator `gainprm` / `biasprm` in the scene XML |

Derive these from the owner YAML and its scene, and reproduce the owner's
resolved per-actuator scale exactly — a scalar owner and a regex-map owner are
not interchangeable. Do not hand-copy joint ranges or gains into a second place
that can silently drift from the asset.

## Hand-off testing

Before integrating policy → safety → motor, test the safety layer in
isolation:

1. Inject a NaN action and verify the command is rejected.
2. Inject an out-of-range joint target and verify clamping uses the joint range
   from the owner's scene XML.
3. Cut the policy feed mid-run and verify the controller enters its configured
   safe state.

## See also

- {doc}`5-onnx_runtime`
- {doc}`9-troubleshooting`
- {doc}`2-g1_whole_body`
