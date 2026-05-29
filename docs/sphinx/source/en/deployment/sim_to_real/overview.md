# Sim-to-Real Overview

This page is the *map* of UniLab's sim-to-real workflow. Every subsequent
page in this section drills into one stage.

## What "sim-to-real" means in UniLab

A deployable UniLab policy is the exported policy plus the exact observation
and action contracts used by the selected task owner. The G1 WBT helper path
materializes this as `policy.onnx`, `deploy_config.yaml`, and a motion binary;
other robots need an equivalent hardware-side runtime that:

1. Reads sensors → assembles the **same observation vector** the policy saw
   in simulation.
2. Runs `policy.onnx` through a runtime that supports the exported graph.
3. Maps the action vector to the same actuator interface used by the env's
   `SimBackend`.

If any of those three things drifts between sim and deployment, debug the
contract first before changing reward or hardware tuning.

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
| Train | Task owner YAML + training script | {doc}`../../user_guide/training/cli_reference` |
| Curriculum + DR | `unilab.dr` + task-side providers | {doc}`domain_randomization` |
| Cross-backend sanity | `task=<task>/<other_backend>` | {doc}`../sim_to_sim/backend_swap` |
| ONNX export | Training playback scripts + deploy helpers | {doc}`onnx_runtime` |
| Latency / obs lag | Task config flags and deploy-side logs | {doc}`latency_budget` |
| Safety layer | Hardware-side clamp / fallback | {doc}`safety_layers` |
| Robot bringup | Robot-specific guides | {doc}`g1_whole_body`, {doc}`go2_locomotion`, {doc}`allegro_inhand` |

## What you should have before starting

::::{admonition} Pre-flight checklist
:class: note

1. A **converged training run** with stable reward AND a stable success
   criterion (motion tracking error, drop count, etc.).
2. The same policy passes evaluation in **both** MuJoCo and Motrix when both
   support the task — if not, you have a backend-dependent reward leak; see
   {doc}`../sim_to_sim/reward_parity`.
3. **Domain randomization** ranges large enough that reward varies smoothly
   when you sweep DR strength — a brittle policy in sim is a brittle policy
   on hardware.
4. **No backend feature leakage** in the env — verify via the developer
   guide's {doc}`../../developer_guide/contracts/backend_contract`.
5. An **observation spec** you can implement on hardware. If your policy
   reads `body_lin_vel`, you need a deploy-side estimator or a task owner
   variant that removes that signal from the actor input.

::::

## The most common failure modes

- **Observation drift.** Sensor pre-processing differs between sim and deploy
  runtime (units, frame, filter cutoffs). Log the first deploy-side
  observation window and compare it with a sim rollout built from the same
  owner YAML.
- **Action latency.** Some task configs expose one-step delayed action
  execution through `control_config.simulate_action_latency`. Measure the
  deploy loop and make the training owner match that contract before a
  hardware run. See {doc}`latency_budget`.
- **Friction / damping mismatch.** Especially for in-hand manipulation.
  Sweep friction in DR; cross-check via {doc}`../sim_to_sim/contact_and_friction_alignment`.
- **Reset transients.** Sim resets to a stable pose; deployment starts from a
  controller state. The safety layer must reject malformed observations and
  unsafe actions before they reach the motor driver.

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
