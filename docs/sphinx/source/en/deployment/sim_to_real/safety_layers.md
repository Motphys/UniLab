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
training script. The repository's G1 helper path exports deploy config and runs
a MuJoCo prototype; it does not implement a production motor-driver safety
loop.

## What the policy assumes you've configured

The G1 deployment helper exports these fields into `deploy_config.yaml`:

```yaml
action_scale: 2.0
ema_alpha: 1.0
default_angles: [...]
joint_lower: [...]
joint_upper: [...]
kp: [...]
kd: [...]
```

`scripts/deploy/sim_prototype.py` consumes the same fields and applies
`action * action_scale + default_angles`, joint clipping, and EMA smoothing.
Hardware controllers should consume generated config rather than hand-copying
joint ranges or gains.

## Hand-off testing

Before integrating policy → safety → motor, test the safety layer in
isolation:

1. Inject a NaN action and verify the command is rejected.
2. Inject an out-of-range joint target and verify clamping uses
   `joint_lower` / `joint_upper` from `deploy_config.yaml`.
3. Cut the policy feed mid-run and verify the controller enters its configured
   safe state.

## See also

- {doc}`onnx_runtime`
- {doc}`troubleshooting`
- {doc}`g1_whole_body`
