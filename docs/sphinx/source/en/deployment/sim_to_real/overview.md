# Sim-to-Real Overview

This page is the *map* of UniLab's sim-to-real workflow. Every subsequent
page in this section drills into one stage.

## What "sim-to-real" means in UniLab

A trained UniLab policy is just `(policy.onnx, observation spec, action spec,
normalization stats)`. The hardware-side code is a thin runtime that:

1. Reads sensors → assembles the **same observation vector** the policy saw
   in simulation.
2. Runs `policy.onnx` (CPU or NPU) on that vector.
3. Maps the action vector to the same actuator interface used by the env's
   `SimBackend`.

If any of those three things drifts between sim and real, the policy fails.
The chapters below address each failure mode in turn.

## End-to-end pipeline

```{mermaid}
flowchart LR
    A[Train in UniLab] --> B[Curriculum + DR]
    B --> C[Validate in alt backend]
    C --> D[Export ONNX]
    D --> E[Latency / lag injection]
    E --> F[Safety layer]
    F --> G[Hardware bringup]
    G --> H[Closed-loop run]
    H -. iterate .-> B
```

| Stage | UniLab artefact | Page |
|---|---|---|
| Train | YAML owner + `train` CLI | {doc}`../../getting_started/training` |
| Curriculum + DR | `unilab.dr` + task-side providers | {doc}`domain_randomization_for_real` |
| Cross-backend sanity | `task=<task>/<other_backend>` | {doc}`../sim_to_sim/why_switch` |
| ONNX export | `unilab.algos.torch.common.ane_wrapper` + scripts | {doc}`onnx_export_and_runtime` |
| Latency / obs lag | DR sensors-side or env wrapper | {doc}`latency_and_observation_lag` |
| Safety layer | Hardware-side clamp / fallback | {doc}`safety_layers` |
| Robot bringup | Robot-specific guides | {doc}`g1_whole_body`, {doc}`go2_locomotion`, {doc}`allegro_inhand` |

## What you should have before starting

::::{admonition} Pre-flight checklist
:class: note

1. A **converged training run** with stable reward AND a stable success
   criterion (motion tracking error, drop count, etc.).
2. The same policy passes evaluation in **both** MuJoCo and Motrix when both
   support the task — if not, you have a backend-dependent reward leak; see
   {doc}`../sim_to_sim/reward_parity_checks`.
3. **Domain randomization** ranges large enough that reward varies smoothly
   when you sweep DR strength — a brittle policy in sim is a brittle policy
   on hardware.
4. **No backend feature leakage** in the env — verify via the developer
   guide's {doc}`../../developer_guide/contracts/backend_capability`.
5. An **observation spec** you can implement on hardware. If your policy
   reads `body_lin_vel`, you need a state estimator on the robot. Don't
   discover this after training for 12 hours.

::::

## The most common failure modes

- **Observation drift.** Sensor pre-processing differs between sim and real
  (units, frame, filter cutoffs). Always log the first 1s of observations
  on real hardware and `diff` against a sim rollout.
- **Action latency.** Sim assumes zero-latency torque commands; real motors
  have ~5–20 ms transport delay plus driver dynamics. Inject latency in sim
  *before* you train. See {doc}`latency_and_observation_lag`.
- **Friction / damping mismatch.** Especially for in-hand manipulation.
  Sweep friction in DR; cross-check via {doc}`../sim_to_sim/contact_and_friction_alignment`.
- **Reset transients.** Sim resets to a stable pose; on the robot you walk
  in. The first 200 ms of inference must be safe even if the observation is
  garbage. Hardware-side safety layer handles this.

## Per-robot quick links

::::{grid} 3
:gutter: 2

:::{grid-item-card} 🤖 G1 whole-body
:link: g1_whole_body
:link-type: doc

Humanoid motion tracking deployment, joint clamp ranges, IMU alignment.
:::

:::{grid-item-card} 🐕 Go2 locomotion
:link: go2_locomotion
:link-type: doc

Joystick + rough terrain policies on Go2 and Go2W.
:::

:::{grid-item-card} ✋ Allegro in-hand
:link: allegro_inhand
:link-type: doc

Dexterous cube reorientation, tactile-free deployment, grasp generator.
:::

::::
