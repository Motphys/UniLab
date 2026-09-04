---
sd_hide_title: true
---

# UniLab Documentation

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab

### Configure task semantics once. Run robot RL across physics backends.

{bdg-primary}`Python >=3.10,<3.14` {bdg-secondary}`Hydra + Manager API` {bdg-info}`Cross-backend contract` {bdg-success}`uv workflow`

UniLab turns task semantics into reusable configuration: assemble manager terms,
select a physics backend, and run the same train/eval workflow on the hardware
available to you. Use this landing page to install, run a first demo, follow it
with a smoke job, choose an algorithm or backend, or jump into deployment and
extension docs.

```{button-ref} 1-getting_started/1-quick_demo
:ref-type: doc
:color: primary
:class: sd-px-4 sd-py-2

Quick Demo
```
```{button-ref} 2-user_guide/0-index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4 sd-py-2

User guide
```
:::

::::

## Why UniLab

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Configure tasks without boilerplate
Compose actions, observations, rewards, terminations, events, commands, and
curricula from manager terms in Hydra owner YAML. Common task variants need no
new environment class.
:::

:::{grid-item-card} Backend choice stays in config
Move between current and future physics adapters with CLI flags such as
`--task go2_joystick_flat --sim motrix`; the CLI composes the matching owner
YAML under `src/unilab/conf/`.
:::

:::{grid-item-card} Scale across hardware
The task contract connects CPU-parallel and external-worker simulation to
accelerator learners, so the experiment can grow with the hardware available
to you.
:::

::::

## Quick Install And Smoke Run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
uv run demo dance
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

For the full README-style walkthrough, see {doc}`1-getting_started/1-quick_demo`.
For platform-specific setup, see {doc}`1-getting_started/2-installation`.

## Start where you are

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Install the repo
:link: 1-getting_started/2-installation
:link-type: doc
Set up `uv`, sync dependencies, and pick the platform profile that matches your
machine.
:::

:::{grid-item-card} Run or replay training
:link: 1-getting_started/1-quick_demo
:link-type: doc
Run a pre-trained demo first, then move to PPO training, evaluation, playback,
or checkpoint resume.
:::

:::{grid-item-card} Choose a physics backend
:link: 2-user_guide/3-backends/0-index
:link-type: doc
Select a backend through task owner YAMLs and read its installation and
capability requirements.
:::

:::{grid-item-card} Pick an algorithm
:link: 2-user_guide/2-algorithms/0-index
:link-type: doc
Compare PPO, APPO, SAC, TD3, FlashSAC, HIM-PPO, and HORA entrypoints.
:::

:::{grid-item-card} Deploy or switch sims
:link: 3-deployment/1-sim_to_real/1-overview
:link-type: doc
Follow sim-to-real checklists or use the sim-to-sim docs to swap MuJoCo and
Motrix.
:::

:::{grid-item-card} Extend safely
:link: 4-developer_guide/0-index
:link-type: doc
Read the env, backend, runner, registry, and task-owner contracts before adding
tasks, backends, algorithms, or terrain.
:::

::::

## Architecture Snapshot

```{mermaid}
flowchart LR
  cli["uv run train/eval<br/>--algo --task --sim"] --> owner["Task owner YAML<br/>src/unilab/conf/*/task/..."]
  cli --> script["Thin script routing<br/>src/unilab/scripts/train_*.py"]
  owner --> registry["Registry bootstrap<br/>src/unilab/base/registry.py"]
  registry --> env["NpEnv contract<br/>obs dict + info dict"]
  env --> backend["SimBackend<br/>unisim-core adapters"]
  env --> factory["EnvFactory contract"]
  factory --> runtime["Runner / IPC<br/>unilab-rl async runtime"]
  runtime --> learner["Learner<br/>PPO / APPO / SAC / TD3"]
```

The load-bearing contracts are documented in
{doc}`4-developer_guide/0-index`; backend support evidence is summarized in
{doc}`2-user_guide/3-backends/0-index`.

## Hardware And Algorithm Coverage

This snapshot only lists coverage backed by checked-in scripts, owner YAMLs, and
the generated support-matrix evidence grades. The repository currently has no
committed benchmark manifest or separate recommendation metadata.

| Robot / task family | Algorithm paths with repo evidence | Backend evidence |
| --- | --- | --- |
| Go1 joystick | PPO, APPO, TD3 | PPO has tested MuJoCo and Motrix rows. APPO has tested MuJoCo rows and Motrix registered rows. TD3 has a Motrix owner YAML for `go1_joystick_flat`. |
| Go2 joystick | PPO, FlashSAC, TD3 | PPO has tested MuJoCo and Motrix rows. FlashSAC has MuJoCo owner YAMLs for `go2_joystick_flat`; TD3 has a Motrix owner YAML for `go2_joystick_flat`. |
| Go2 arm manip-loco | PPO, HIM-PPO | Committed MuJoCo owner YAMLs are present under `src/unilab/conf/ppo/task/go2_arm_manip_loco/` and `src/unilab/conf/ppo_him/task/go2_arm_manip_loco/`. |
| Go2W joystick | PPO | PPO owner YAMLs exist for MuJoCo and Motrix flat/rough variants under `src/unilab/conf/ppo/task/go2w_joystick_*`. |
| G1 locomotion / tracking | PPO, APPO, SAC, TD3 | PPO, APPO, and SAC include committed MuJoCo and Motrix owner YAMLs for G1 tasks; TD3 has a `g1_walk_flat` MuJoCo owner. |
| Allegro in-hand | PPO, APPO | PPO and APPO have committed MuJoCo and Motrix owner YAMLs for Allegro in-hand tasks. |
| Sharpa in-hand | PPO, APPO HORA teacher, HORA distillation | Sharpa owner YAMLs are committed for PPO/APPO teacher paths; student distillation uses `src/unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml`. |

```{toctree}
:hidden:
:caption: Documentation

1-getting_started/0-index
2-user_guide/0-index
3-deployment/0-index
4-developer_guide/0-index
5-reference/0-index
```
