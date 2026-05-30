# Deployment

A hands-on playbook for moving a UniLab policy across hardware, simulation
backends, and source frameworks. Each tutorial follows the same shape:

1. **What you start with** — the trained artefact and config.
2. **What changes** — the minimal set of edits in code, YAML, and assets.
3. **How you validate** — concrete commands and checkpoints.

## Choose your journey

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🤖 Sim → Real
:link: sim_to_real/index
:link-type: doc
:class-card: sd-shadow-md

Prepare a trained policy for G1 / Go2 / Allegro bring-up with ONNX exports and
deploy-side contract checks.
:::

:::{grid-item-card} 🔀 Sim → Sim
:link: sim_to_sim/index
:link-type: doc
:class-card: sd-shadow-md

Switch the same task between MuJoCo and Motrix without retraining from scratch.
:::

:::{grid-item-card} 🔁 Framework Migration
:link: framework_migration/index
:link-type: doc
:class-card: sd-shadow-md

Bring tasks over from Isaac Lab / Legged Gym / rsl_rl / skrl.
:::

::::

---

## 🤖 Sim → Real

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🗺 Overview & pre-flight
:link: sim_to_real/overview
:link-type: doc
End-to-end pipeline + go/no-go checklist.
:::

:::{grid-item-card} 🦿 G1 whole-body
:link: sim_to_real/g1_whole_body
:link-type: doc
29-DoF humanoid; motion-tracking deploy.
:::

:::{grid-item-card} 🐕 Go2 locomotion
:link: sim_to_real/go2_locomotion
:link-type: doc
Joystick, rough terrain, Go2W wheels.
:::

:::{grid-item-card} 🤚 Allegro in-hand
:link: sim_to_real/allegro_inhand
:link-type: doc
Cube rotation; friction + vision.
:::

:::{grid-item-card} 📦 ONNX export & runtime
:link: sim_to_real/onnx_runtime
:link-type: doc
Training playback exports, ONNX Runtime checks, and deploy prototype inputs.
:::

:::{grid-item-card} 🎲 Sim-to-real DR
:link: sim_to_real/domain_randomization
:link-type: doc
Priority-ordered DR recipes.
:::

:::{grid-item-card} 🛡 Safety layers
:link: sim_to_real/safety_layers
:link-type: doc
Soft limits, EMA, e-stop, watchdog.
:::

:::{grid-item-card} ⏱ Latency & observation lag
:link: sim_to_real/latency_budget
:link-type: doc
Training-side latency knobs and deploy-side measurement checks.
:::

:::{grid-item-card} 🔧 Troubleshooting
:link: sim_to_real/troubleshooting
:link-type: doc
Symptom → cause → fix cookbook.
:::

::::

---

## 🔀 Sim → Sim (MuJoCo ↔ Motrix)

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🤔 Backend swap
:link: sim_to_sim/backend_swap
:link-type: doc
:::

:::{grid-item-card} 📝 Owner YAML swap
:link: sim_to_sim/owner_yaml_swap
:link-type: doc
:::

:::{grid-item-card} 🔬 Contact & friction alignment
:link: sim_to_sim/contact_and_friction_alignment
:link-type: doc
:::

:::{grid-item-card} ⚖ Reward parity checks
:link: sim_to_sim/reward_parity
:link-type: doc
:::

:::{grid-item-card} 🎞 Playback differences
:link: sim_to_sim/playback_and_snapshot_differences
:link-type: doc
:::

:::{grid-item-card} 🚫 Known capability gaps
:link: sim_to_sim/capability_gaps
:link-type: doc
:::

::::

---

## 🔁 Framework Migration

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} From **Isaac Lab**
:link: framework_migration/from_isaac_lab
:link-type: doc
GPU-resident → CPU + shared-mem.
:::

:::{grid-item-card} From **Legged Gym**
:link: framework_migration/from_legged_gym
:link-type: doc
Class-based env → NpEnv.
:::

:::{grid-item-card} From **rsl_rl**
:link: framework_migration/from_rsl_rl
:link-type: doc
Trainer split: collector + learner.
:::

:::{grid-item-card} From **skrl**
:link: framework_migration/from_skrl
:link-type: doc
Algo coverage and trade-offs.
:::

:::{grid-item-card} 📋 Config translation cheatsheet
:link: framework_migration/task_config_translation
:link-type: doc
Side-by-side field map.
:::

:::{grid-item-card} 📒 Reward porting cookbook
:link: framework_migration/reward_porting
:link-type: doc
Common reward terms in UniLab style.
:::

::::

```{toctree}
:hidden:
:caption: Deployment

sim_to_real/index
sim_to_sim/index
framework_migration/index
```
