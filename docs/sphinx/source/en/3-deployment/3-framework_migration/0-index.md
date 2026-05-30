# Framework Migration

Bring existing tasks or training flows from adjacent RL frameworks into UniLab's
contract-driven layout.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} From Isaac Lab
:link: 1-from_isaac_lab
:link-type: doc
Map GPU-resident task structure to UniLab's CPU sim and learner split.
:::

:::{grid-item-card} From Legged Gym
:link: 2-from_legged_gym
:link-type: doc
Move class-based environments into the `NpEnv` contract.
:::

:::{grid-item-card} From RSL-RL
:link: 3-from_rsl_rl
:link-type: doc
Separate trainer assumptions from UniLab runner assembly.
:::

:::{grid-item-card} From skrl
:link: 4-from_skrl
:link-type: doc
Map algorithm entrypoints and config ownership.
:::

:::{grid-item-card} Config translation
:link: 5-task_config_translation
:link-type: doc
Compare common field ownership across configs.
:::

:::{grid-item-card} Reward porting
:link: 6-reward_porting
:link-type: doc
Port reward terms without breaking env/backend contracts.
:::

::::

```{toctree}
:hidden:

1-from_isaac_lab
2-from_legged_gym
3-from_rsl_rl
4-from_skrl
5-task_config_translation
6-reward_porting
```
