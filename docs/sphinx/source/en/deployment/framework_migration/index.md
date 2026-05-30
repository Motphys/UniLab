# Framework Migration

Bring existing tasks or training flows from adjacent RL frameworks into UniLab's
contract-driven layout.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} From Isaac Lab
:link: from_isaac_lab
:link-type: doc
Map GPU-resident task structure to UniLab's CPU sim and learner split.
:::

:::{grid-item-card} From Legged Gym
:link: from_legged_gym
:link-type: doc
Move class-based environments into the `NpEnv` contract.
:::

:::{grid-item-card} From RSL-RL
:link: from_rsl_rl
:link-type: doc
Separate trainer assumptions from UniLab runner assembly.
:::

:::{grid-item-card} From skrl
:link: from_skrl
:link-type: doc
Map algorithm entrypoints and config ownership.
:::

:::{grid-item-card} Config translation
:link: task_config_translation
:link-type: doc
Compare common field ownership across configs.
:::

:::{grid-item-card} Reward porting
:link: reward_porting
:link-type: doc
Port reward terms without breaking env/backend contracts.
:::

::::

```{toctree}
:hidden:

from_isaac_lab
from_legged_gym
from_rsl_rl
from_skrl
task_config_translation
reward_porting
```
