# Sim-to-Sim

Move the same task between MuJoCo and Motrix through owner YAMLs and backend
contracts.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Backend swap
:link: backend_swap
:link-type: doc
Select the backend through the task owner YAML.
:::

:::{grid-item-card} Owner YAML swap
:link: owner_yaml_swap
:link-type: doc
Add or inspect a backend YAML for an existing task.
:::

:::{grid-item-card} Contact and friction
:link: contact_and_friction_alignment
:link-type: doc
Align contact-related assumptions across simulators.
:::

:::{grid-item-card} Reward parity
:link: reward_parity
:link-type: doc
Check reward terms near backend boundaries.
:::

:::{grid-item-card} Playback differences
:link: playback_and_snapshot_differences
:link-type: doc
Understand renderer and snapshot capability differences.
:::

:::{grid-item-card} Capability gaps
:link: capability_gaps
:link-type: doc
Document unsupported backend features with evidence.
:::

::::

```{toctree}
:hidden:

backend_swap
owner_yaml_swap
contact_and_friction_alignment
reward_parity
playback_and_snapshot_differences
capability_gaps
```
