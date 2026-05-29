---
sd_hide_title: true
---

# UniLab Documentation

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab

### Contract-driven robot learning infrastructure for CPU simulation and accelerator learning.

{bdg-primary}`Python >=3.10,<3.14` {bdg-secondary}`Hydra owner YAML` {bdg-info}`MuJoCo + Motrix` {bdg-success}`uv workflow`

UniLab routes robot RL through explicit training scripts, task-owner Hydra
configs, and backend contracts. Use the landing page to install, run a smoke
training job, choose an algorithm/backend, or jump into deployment and extension
docs.

```{button-ref} getting_started/quickstart
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

## Why UniLab

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} CPU simulation, accelerator learning
The README describes UniLab as CPU physics simulation connected to policy
training through shared memory, with MuJoCo and Motrix as simulation backends.
:::

:::{grid-item-card} Backend choice stays in config
Switch backends through `task=<task>/<backend>` owner YAMLs under `conf/`; do
not use `training.sim_backend` as a standalone backend switch.
:::

:::{grid-item-card} Deployment paths are documented
The deployment docs cover sim-to-real, sim-to-sim, ONNX/runtime export, safety
layers, and robot-specific notes for G1, Go2, and Allegro.
:::

::::

## Quick Install And Smoke Run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

For platform-specific setup, see {doc}`getting_started/installation`.
For the longer first-run walkthrough, see
{doc}`getting_started/quickstart`.

## Start where you are

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Install the repo
:link: getting_started/installation
:link-type: doc
Set up `uv`, sync dependencies, and pick the platform profile that matches your
machine.
:::

:::{grid-item-card} Run or replay training
:link: getting_started/quickstart
:link-type: doc
Start with PPO on Go2, then move to evaluation, playback, or checkpoint resume.
:::

:::{grid-item-card} Choose a backend
:link: user_guide/backends/choosing_a_backend
:link-type: doc
Compare MuJoCo and Motrix through task owner YAMLs and backend capability docs.
:::

:::{grid-item-card} Pick an algorithm
:link: user_guide/algorithms/overview
:link-type: doc
Compare PPO, APPO, SAC, TD3, FlashSAC, MLX PPO, HIM-PPO, and HORA entrypoints.
:::

:::{grid-item-card} Deploy or switch sims
:link: deployment/sim_to_real/overview
:link-type: doc
Follow sim-to-real checklists or use the sim-to-sim docs to swap MuJoCo and
Motrix.
:::

:::{grid-item-card} Extend safely
:link: developer_guide/index
:link-type: doc
Read the env, backend, runner, registry, and task-owner contracts before adding
tasks, backends, algorithms, or terrain.
:::

::::

## Architecture Snapshot

```{mermaid}
flowchart LR
  owner["Task owner YAML<br/>conf/*/task/..."] --> script["uv run scripts/train_*.py"]
  owner --> registry["Registry bootstrap<br/>src/unilab/base/registry.py"]
  registry --> env["NpEnv contract<br/>obs dict + info dict"]
  env --> backend["SimBackend<br/>MuJoCo or Motrix"]
  env --> runtime["Runner / IPC<br/>shared memory lifecycle"]
  runtime --> learner["Learner<br/>PPO / APPO / SAC / TD3 / MLX"]
```

The load-bearing contracts are documented in
{doc}`developer_guide/index`; backend support evidence is summarized in
{doc}`user_guide/backends/index`.

## Hardware And Algorithm Coverage

This snapshot only lists coverage backed by checked-in scripts, owner YAMLs, and
the generated support-matrix evidence grades. The repository currently has no
committed benchmark manifest or separate recommendation metadata.

| Robot / task family | Algorithm paths with repo evidence | Backend evidence |
| --- | --- | --- |
| Go1 joystick | PPO (torch, MLX), APPO, TD3 | PPO has tested MuJoCo and Motrix rows. APPO has tested MuJoCo rows and Motrix registered rows. TD3 has a Motrix owner YAML for `go1_joystick_flat`. |
| Go2 joystick / handstand | PPO (torch, MLX), FlashSAC, TD3 | PPO has tested MuJoCo and Motrix rows. FlashSAC has MuJoCo owner YAMLs for `go2_joystick_flat`; TD3 has a Motrix owner YAML for `go2_joystick_flat`. |
| Go2 arm manip-loco | PPO, HIM-PPO | Committed MuJoCo owner YAMLs are present under `conf/ppo/task/go2_arm_manip_loco/` and `conf/ppo_him/task/go2_arm_manip_loco/`. |
| Go2W joystick | PPO (torch, MLX configured) | PPO owner YAMLs exist for MuJoCo and Motrix flat/rough variants under `conf/ppo/task/go2w_joystick_*`. |
| G1 locomotion / tracking | PPO (torch, MLX), APPO, SAC, TD3 | PPO, APPO, and SAC include committed MuJoCo and Motrix owner YAMLs for G1 tasks; TD3 has a `g1_walk_flat` MuJoCo owner. |
| Allegro in-hand | PPO (torch, MLX configured), APPO | PPO and APPO have committed MuJoCo and Motrix owner YAMLs for Allegro in-hand tasks. |
| Sharpa in-hand | PPO, APPO HORA teacher, HORA distillation | Sharpa owner YAMLs are committed for PPO/APPO teacher paths; student distillation uses `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`. |

```{toctree}
:hidden:
:caption: Documentation

getting_started/index
user_guide/index
deployment/index
developer_guide/index
reference/index
```
