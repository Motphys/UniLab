# Sim-to-Real

Prepare a trained UniLab policy for hardware bring-up. Start with the overview,
then move through export, randomization, safety, latency, and robot-specific
notes.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview and pre-flight
:link: 1-overview
:link-type: doc
End-to-end pipeline and go/no-go checklist.
:::

:::{grid-item-card} ONNX runtime
:link: 5-onnx_runtime
:link-type: doc
Training playback exports, ONNX Runtime checks, and deploy inputs.
:::

:::{grid-item-card} Domain randomization
:link: 6-domain_randomization
:link-type: doc
Priority-ordered randomization checks for real-world transfer.
:::

:::{grid-item-card} Safety layers
:link: 7-safety_layers
:link-type: doc
Soft limits, action filters, watchdogs, and e-stop boundaries.
:::

:::{grid-item-card} Latency budget
:link: 8-latency_budget
:link-type: doc
Training-side latency knobs and deploy-side measurement checks.
:::

:::{grid-item-card} Troubleshooting
:link: 9-troubleshooting
:link-type: doc
Symptom, cause, and fix notes for hardware bring-up.
:::

:::{grid-item-card} G1 whole-body
:link: 2-g1_whole_body
:link-type: doc
Motion-tracking deployment notes for the G1 path.
:::

:::{grid-item-card} Go2 locomotion
:link: 3-go2_locomotion
:link-type: doc
Joystick, rough terrain, and Go2W deployment notes.
:::

:::{grid-item-card} Allegro in-hand
:link: 4-allegro_inhand
:link-type: doc
In-hand manipulation deployment checks.
:::

::::

```{toctree}
:hidden:

1-overview
2-g1_whole_body
3-go2_locomotion
4-allegro_inhand
5-onnx_runtime
6-domain_randomization
7-safety_layers
8-latency_budget
9-troubleshooting
```
