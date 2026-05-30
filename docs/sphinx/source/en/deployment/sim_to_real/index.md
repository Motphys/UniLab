# Sim-to-Real

Prepare a trained UniLab policy for hardware bring-up. Start with the overview,
then move through export, randomization, safety, latency, and robot-specific
notes.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview and pre-flight
:link: overview
:link-type: doc
End-to-end pipeline and go/no-go checklist.
:::

:::{grid-item-card} ONNX runtime
:link: onnx_runtime
:link-type: doc
Training playback exports, ONNX Runtime checks, and deploy inputs.
:::

:::{grid-item-card} Domain randomization
:link: domain_randomization
:link-type: doc
Priority-ordered randomization checks for real-world transfer.
:::

:::{grid-item-card} Safety layers
:link: safety_layers
:link-type: doc
Soft limits, action filters, watchdogs, and e-stop boundaries.
:::

:::{grid-item-card} Latency budget
:link: latency_budget
:link-type: doc
Training-side latency knobs and deploy-side measurement checks.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc
Symptom, cause, and fix notes for hardware bring-up.
:::

:::{grid-item-card} G1 whole-body
:link: g1_whole_body
:link-type: doc
Motion-tracking deployment notes for the G1 path.
:::

:::{grid-item-card} Go2 locomotion
:link: go2_locomotion
:link-type: doc
Joystick, rough terrain, and Go2W deployment notes.
:::

:::{grid-item-card} Allegro in-hand
:link: allegro_inhand
:link-type: doc
In-hand manipulation deployment checks.
:::

::::

```{toctree}
:hidden:

overview
g1_whole_body
go2_locomotion
allegro_inhand
onnx_runtime
domain_randomization
safety_layers
latency_budget
troubleshooting
```
