---
sd_hide_title: true
---

# UniLab Documentation

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab

### Contract-driven robot learning infrastructure for MuJoCo and Motrix.

UniLab keeps task ownership in Hydra config, isolates backend-specific behavior
behind `SimBackend`, and routes training through explicit scripts. Start with a
known task, then move to backend selection, algorithms, transfer, or extension
work as needed.

```{button-ref} user_guide/getting_started/quickstart
:ref-type: doc
:color: primary
:class: sd-px-4 sd-py-2

Quickstart
```
```{button-ref} user_guide/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4 sd-py-2

User guide
```
:::

::::

## Start where you are

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} First training run
:link: user_guide/getting_started/quickstart
:link-type: doc
Install with `uv`, run PPO on Go2, and check the run output.
:::

:::{grid-item-card} Backend selection
:link: user_guide/backends/choosing_a_backend
:link-type: doc
Choose MuJoCo or Motrix through the task owner YAML.
:::

:::{grid-item-card} Algorithms
:link: user_guide/algorithms/overview
:link-type: doc
Compare PPO, APPO, off-policy learners, MLX PPO, HIM-PPO, and HORA.
:::

:::{grid-item-card} Sim-to-real
:link: transfer/sim_to_real/overview
:link-type: doc
Review deployment contracts, observation statistics, and safety layers.
:::

:::{grid-item-card} API reference

{{ api_ref_blurb }}

{{ api_ref_button }}
:::

:::{grid-item-card} Extending UniLab
:link: developer_guide/index
:link-type: doc
Read the env, backend, runner, registry, and task owner contracts.
:::

::::

## Quick install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
```

For platform-specific dependency paths, see
{doc}`user_guide/getting_started/installation`.

```{toctree}
:hidden:
:caption: Documentation

user_guide/index
user_guide/getting_started/installation
user_guide/getting_started/quickstart
user_guide/getting_started/training
user_guide/getting_started/configuration_overrides
user_guide/getting_started/evaluation_and_playback
user_guide/backends/index
user_guide/algorithms/overview
user_guide/tasks/g1_motion_tracking
user_guide/domain_randomization/index
user_guide/terrain/procedural
user_guide/manipulation/dexterous_inhand
user_guide/tooling/wandb_and_tensorboard
transfer/index
transfer/sim_to_real/overview
transfer/sim_to_sim/why_switch
transfer/framework_migration/from_isaac_lab
developer_guide/index
developer_guide/contributing
developer_guide/contributing_workflow
agents/index
/adr/README
/adr/ADR-0000-index
/api_reference/index
/api_reference/base/index
/api_reference/envs/index
/api_reference/algos/index
/api_reference/backend/index
/api_reference/training/index
/api_reference/dr/index
/api_reference/top_level
/glossary
/changelog
```
