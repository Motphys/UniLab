# Sim-to-Sim

Move the same task between MuJoCo and Motrix through owner YAMLs and backend
contracts.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Backend swap
:link: 1-backend_swap
:link-type: doc
Select the backend through the task owner YAML.
:::

:::{grid-item-card} Owner YAML swap
:link: 2-owner_yaml_swap
:link-type: doc
Add or inspect a backend YAML for an existing task.
:::

:::{grid-item-card} Contact and friction
:link: 3-contact_and_friction_alignment
:link-type: doc
Align contact-related assumptions across simulators.
:::

:::{grid-item-card} Reward parity
:link: 4-reward_parity
:link-type: doc
Check reward terms near backend boundaries.
:::

:::{grid-item-card} Playback differences
:link: 5-playback_and_snapshot_differences
:link-type: doc
Understand renderer and snapshot capability differences.
:::

:::{grid-item-card} Capability gaps
:link: 6-capability_gaps
:link-type: doc
Document unsupported backend features with evidence.
:::

::::

```{toctree}
:hidden:

1-backend_swap
2-owner_yaml_swap
3-contact_and_friction_alignment
4-reward_parity
5-playback_and_snapshot_differences
6-capability_gaps
```
