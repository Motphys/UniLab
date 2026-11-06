# How-To Guides

End-to-end walkthroughs for common tasks — each step carries the command that
verifies it, so you can confirm the state of your checkout as you go.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Onboard a new robot
:link: 1-onboard_new_robot
:link-type: doc
From URDF/MJCF assets to a first PPO training run, one verified step at a time.
:::

:::{grid-item-card} Add a custom reward term
:link: 2-custom_reward
:link-type: doc
From a function definition to the `reward/<term>` key in the training log.
:::

:::{grid-item-card} Diagnose training problems
:link: 3-training_diagnostics
:link-type: doc
NaN guard dumps, reward anomalies, and backend-mismatch playbooks.
:::

::::

## Reference Quick Links

- Built-in reward / observation / termination terms:
  {doc}`../5-reference/6-manager_terms/0-index`
- Domain randomization fields:
  {doc}`../2-user_guide/5-domain_randomization/2-field_reference`
- Algorithm pages with per-algorithm config reference tables:
  {doc}`../2-user_guide/2-algorithms/0-index`

```{toctree}
:hidden:

1-onboard_new_robot
2-custom_reward
3-training_diagnostics
```
