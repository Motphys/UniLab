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
:link: 1-sim_to_real/0-index
:link-type: doc
:class-card: sd-shadow-md

Prepare a trained policy for G1 / Go2 / Allegro bring-up with ONNX exports and
deploy-side contract checks.
:::

:::{grid-item-card} 🔀 Sim → Sim
:link: 2-sim_to_sim/0-index
:link-type: doc
:class-card: sd-shadow-md

Switch the same task between MuJoCo and Motrix without retraining from scratch.
:::

:::{grid-item-card} 🔁 Framework Migration
:link: 3-framework_migration/0-index
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
:link: 1-sim_to_real/1-overview
:link-type: doc
End-to-end pipeline + go/no-go checklist.
:::

:::{grid-item-card} 🦿 G1 whole-body
:link: 1-sim_to_real/2-g1_whole_body
:link-type: doc
29-DoF humanoid; motion-tracking deploy.
:::

:::{grid-item-card} 🐕 Go2 locomotion
:link: 1-sim_to_real/3-go2_locomotion
:link-type: doc
Joystick, rough terrain, Go2W wheels.
:::

:::{grid-item-card} 🤚 Allegro in-hand
:link: 1-sim_to_real/4-allegro_inhand
:link-type: doc
Cube rotation; friction + vision.
:::

:::{grid-item-card} 📦 ONNX export & runtime
:link: 1-sim_to_real/5-onnx_runtime
:link-type: doc
Training playback exports, ONNX Runtime checks, and deploy prototype inputs.
:::

:::{grid-item-card} 🎲 Sim-to-real DR
:link: 1-sim_to_real/6-domain_randomization
:link-type: doc
Priority-ordered DR recipes.
:::

:::{grid-item-card} 🛡 Safety layers
:link: 1-sim_to_real/7-safety_layers
:link-type: doc
Soft limits, EMA, e-stop, watchdog.
:::

:::{grid-item-card} ⏱ Latency & observation lag
:link: 1-sim_to_real/8-latency_budget
:link-type: doc
Training-side latency knobs and deploy-side measurement checks.
:::

:::{grid-item-card} 🔧 Troubleshooting
:link: 1-sim_to_real/9-troubleshooting
:link-type: doc
Symptom → cause → fix cookbook.
:::

::::

---

## 🔀 Sim → Sim (MuJoCo ↔ Motrix)

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🤔 Backend swap
:link: 2-sim_to_sim/1-backend_swap
:link-type: doc
:::

:::{grid-item-card} 📝 Owner YAML swap
:link: 2-sim_to_sim/2-owner_yaml_swap
:link-type: doc
:::

:::{grid-item-card} 🔬 Contact & friction alignment
:link: 2-sim_to_sim/3-contact_and_friction_alignment
:link-type: doc
:::

:::{grid-item-card} ⚖ Reward parity checks
:link: 2-sim_to_sim/4-reward_parity
:link-type: doc
:::

:::{grid-item-card} 🎞 Playback differences
:link: 2-sim_to_sim/5-playback_and_snapshot_differences
:link-type: doc
:::

:::{grid-item-card} 🚫 Known capability gaps
:link: 2-sim_to_sim/6-capability_gaps
:link-type: doc
:::

::::

---

## 🔁 Framework Migration

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} From **Isaac Lab**
:link: 3-framework_migration/1-from_isaac_lab
:link-type: doc
GPU-resident → CPU + shared-mem.
:::

:::{grid-item-card} From **Legged Gym**
:link: 3-framework_migration/2-from_legged_gym
:link-type: doc
Class-based env → NpEnv.
:::

:::{grid-item-card} From **rsl_rl**
:link: 3-framework_migration/3-from_rsl_rl
:link-type: doc
Trainer split: collector + learner.
:::

:::{grid-item-card} From **skrl**
:link: 3-framework_migration/4-from_skrl
:link-type: doc
Algo coverage and trade-offs.
:::

:::{grid-item-card} 📋 Config translation cheatsheet
:link: 3-framework_migration/5-task_config_translation
:link-type: doc
Side-by-side field map.
:::

:::{grid-item-card} 📒 Reward porting cookbook
:link: 3-framework_migration/6-reward_porting
:link-type: doc
Common reward terms in UniLab style.
:::

::::

```{toctree}
:hidden:
:caption: Deployment

1-sim_to_real/0-index
2-sim_to_sim/0-index
3-framework_migration/0-index
```
