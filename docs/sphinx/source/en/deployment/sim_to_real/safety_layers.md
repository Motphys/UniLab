# Hardware Safety Layers

The policy assumes its actions are safe-by-training. On real hardware that
assumption fails on the first comms glitch. The safety layer is the thin
shim that lives **between the policy output and the motor driver** and
catches the failure modes.

## Required components

```{list-table}
:header-rows: 1
:widths: 30 70

* - Layer
  - Responsibility
* - Schema check
  - Action has correct dtype, shape, finite values. Reject NaN / Inf.
* - Range clamp
  - Hard-clamp each joint target to the URDF limits. Driver-side, not
    Python-side.
* - Δ clamp
  - Reject Δ > policy-trained max per step. Catches single-frame spikes.
* - Rate limit
  - Slew-rate limit applied AFTER clamp.
* - Watchdog
  - If no fresh action arrives within Δt_max, hold the last action; after
    a second Δt_max, fade-to-zero / safe pose.
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

Critically, the safety layer is **not** in Python. Python GC pauses can
exceed 100 ms; that's a guaranteed face-plant on a humanoid. Implement in
C++ or Rust with bounded allocation.

## What the policy assumes you've configured

UniLab task owners declare these in YAML (see
`conf/<task>/owner.yaml`):

```yaml
action_limits:
  joint_pos_min: [-2.0, -1.5, ...]   # URDF lower bounds
  joint_pos_max: [ 2.0,  1.5, ...]
  joint_delta_max: [0.3, 0.3, ...]   # per-step delta cap

safety_pose:
  joint_pos: [0.0, 0.7, -1.4, ...]   # the "sit" / "fold" pose
  ramp_time_s: 0.5
```

Your safety layer should consume these directly — don't hand-copy values.

## Hand-off testing

Before integrating policy → safety → motor, test the safety layer in
isolation:

1. Inject NaN action → verify reject + watchdog engages.
2. Inject Δ = 1.0 rad single-step → verify Δ-clamp limits to 0.3.
3. Cut policy feed mid-run → verify hold-then-fade.

## See also

- {doc}`onnx_export_and_runtime`
- {doc}`troubleshooting`
- {doc}`g1_whole_body`
