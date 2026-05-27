---
sd_hide_title: true
---

# UniLab Documentation

::::{div} landing-hero

:::{div} landing-hero-text

```{image} https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white
:alt: Python 3.10+
:class: landing-badge
```
```{image} https://img.shields.io/badge/license-Apache--2.0-success
:alt: Apache-2.0
:class: landing-badge
```
```{image} https://img.shields.io/badge/backend-MuJoCo%20%7C%20Motrix-orange
:alt: Backends
:class: landing-badge
```
```{image} https://img.shields.io/badge/learner-PyTorch%20%7C%20MLX-EE4C2C
:alt: Learners
:class: landing-badge
```
```{image} https://img.shields.io/badge/deploy-ONNX%20%7C%20CoreML-blue
:alt: Deploy
:class: landing-badge
```

# UniLab

### A Universal Lab for Robot Learning <br/>— *train policies without a GPU simulator.*

UniLab decouples **CPU physics**, **shared-memory IPC**, and **GPU learning**
so the same task config trains on a MacBook, a CUDA server, an AMD ROCm box,
or an Intel Arc workstation — and deploys to real Unitree, Allegro, and
Sharpa hardware through ONNX or Core ML.

```{button-ref} user_guide/getting_started/quickstart
:ref-type: doc
:color: primary
:class: sd-px-4 sd-py-2

Get started → 3 minutes
```
```{button-ref} transfer/sim_to_real/overview
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4 sd-py-2

Sim-to-Real Playbook
```
```{button-link} https://github.com/unilabsim/UniLab
:color: secondary
:outline:
:class: sd-px-4 sd-py-2

⭐ GitHub
```

:::

::::

---

## Why UniLab

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-text-center

:::{grid-item-card} 🧠 No-GPU Sim
:class-card: sd-shadow-sm
The physics step never touches your GPU. CUDA, MPS, ROCm, and XPU all get the
same env and the same algorithm code.
:::

:::{grid-item-card} 🔌 Multi-backend
:class-card: sd-shadow-sm
**MuJoCo** and **Motrix** share one env / runner contract. Backend choice
lives in YAML — Python code stays backend-agnostic.
:::

:::{grid-item-card} 🤖 Real-hardware ready
:class-card: sd-shadow-sm
First-class **G1**, **Go2 / Go2W**, **Allegro**, **Sharpa** deployment with
ONNX export, latency wrappers, and explicit safety layers.
:::

::::

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-text-center

:::{grid-item-card} 📜 Contract-first
:class-card: sd-shadow-sm
Env, backend, runner, and DR contracts are *codified*. PR-level changes can't
silently break downstream tasks.
:::

:::{grid-item-card} ⚡ Cold-path discipline
:class-card: sd-shadow-sm
Hot loops never parse XML or `getattr` into a backend. Borrowed from
production real-time robotics stacks.
:::

:::{grid-item-card} 🎛 Curated algorithms
:class-card: sd-shadow-sm
PPO, APPO, Fast-SAC, Fast-TD3, Flash-SAC, HIM-PPO, HORA — plus an MLX-native
PPO for Apple Silicon.
:::

::::

---

## Start where you are

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 I'm new — give me a working policy
:link: user_guide/getting_started/quickstart
:link-type: doc

Install with `uv`, launch PPO on Go2 in one command, watch reward go up.
+++
**3 minutes** · CPU-only ok
:::

:::{grid-item-card} 🧪 I'm prototyping a new task
:link: user_guide/index
:link-type: doc

Backends, algorithms, task zoo, domain randomization, terrains, tooling —
the full user manual.
+++
The **User Guide**
:::

:::{grid-item-card} 🤖 I'm taking a policy to a real robot
:link: transfer/sim_to_real/overview
:link-type: doc

Pre-flight checklists, ONNX export, latency budgets, safety layers, and per-
robot bring-up guides (G1, Go2, Allegro).
+++
**Sim-to-Real**
:::

:::{grid-item-card} 🔁 I'm porting from Isaac Lab / Legged Gym
:link: transfer/framework_migration/from_isaac_lab
:link-type: doc

Concept-by-concept mapping, config translation cheatsheet, and a reward
porting cookbook.
+++
**Framework Migration**
:::

:::{grid-item-card} 🧩 I need the precise API surface

{{ api_ref_blurb }}

{{ api_ref_button }}
+++
**{{ api_ref_label }}**
:::

:::{grid-item-card} 🛠 I'm extending UniLab
:link: developer_guide/index
:link-type: doc

Architecture, contracts (NpEnv / SimBackend / RunnerLifecycle / DR), ADRs,
and contribution workflow.
+++
**Developer Guide**
:::

::::

---

## Architecture at a glance

```{mermaid}
flowchart LR
    subgraph CPU["💻 CPU Workers"]
        direction TB
        S1["MuJoCo / Motrix<br/>physics step"]
        S2["NpEnv<br/>obs · reward · done"]
        S3["DR manager<br/>actuator · friction · push"]
        S1 --> S2 --> S3
    end

    subgraph IPC["🔗 Shared-memory IPC"]
        direction TB
        R["RolloutRingBuffer"]
        B["ReplayBuffer"]
    end

    subgraph GPU["🚀 GPU / Accel Learner"]
        direction TB
        A["PPO / APPO<br/>SAC · TD3"]
        N["torch.nn / mlx.nn"]
        A --> N
    end

    CPU -- "experience" --> IPC
    IPC -- "minibatches" --> GPU
    GPU -- "actor weights" --> CPU

    classDef cpu fill:#fef3c7,stroke:#92400e,color:#451a03;
    classDef ipc fill:#ddd6fe,stroke:#5b21b6,color:#1e1b4b;
    classDef gpu fill:#dcfce7,stroke:#15803d,color:#052e16;
    class S1,S2,S3 cpu;
    class R,B ipc;
    class A,N gpu;
```

Same diagram, in words:

1. **CPU workers** run `mujoco` or `motrix` and emit a NumPy `state` batch.
2. The **shared-memory rings** copy that batch into a tensor view the learner
   can read without an extra serialization hop.
3. The **GPU learner** does the gradient update and writes new actor weights
   back to the shared region — workers pick them up at the next reset.

Read more in {doc}`developer_guide/architecture/index`.

---

## Hardware & algorithm coverage

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🤖 Robots
:class-card: sd-shadow-sm

| Robot | DoF | Tasks |
|---|---|---|
| Unitree **G1** | 29 | locomotion · motion-tracking · whole-body |
| Unitree **Go1 / Go2 / Go2W** | 12 + wheels | joystick · rough terrain · arm-mounted |
| **Allegro** hand | 16 | in-hand cube rotation |
| **Sharpa** hand | 20 | in-hand cube rotation · grasp gen |
| **Go2 + Airbot** | 12 + 6 | manip-loco |
:::

:::{grid-item-card} 🎛 Algorithms
:class-card: sd-shadow-sm

| Family | Variants |
|---|---|
| **On-policy** | PPO · APPO · HIM-PPO · HORA |
| **Off-policy** | Fast-SAC · Fast-TD3 · Flash-SAC |
| **Apple Silicon** | MLX-PPO (native `mlx.nn`) |
| **Deploy** | ONNX (PyTorch) · CoreML (MLX) |
:::

::::

---

## Quick install

```bash
# 1. uv is the recommended package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone + sync (Motrix backend; use --extra mujoco if preferred)
git clone https://github.com/unilabsim/UniLab.git && cd UniLab
uv sync --extra motrix

# 3. first training run — Go2 joystick on flat terrain
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

For ROCm / Intel Arc / cluster setups see
{doc}`user_guide/getting_started/installation`.

---

## Recent highlights

```{div} feature-list

- **2026-05** · Whole-body motion tracking on Unitree G1 with ONNX deploy
  path → {doc}`transfer/sim_to_real/g1_whole_body`
- **2026-04** · MLX-PPO on Apple Silicon, end-to-end ANE deployment →
  {doc}`user_guide/algorithms/mlx_ppo`
- **2026-03** · Motrix backend reaches feature parity with MuJoCo on
  locomotion tasks → {doc}`user_guide/backends/choosing_a_backend`
- **2026-02** · Domain randomization contract v2 — declarative YAML, no
  more env-side `self.cfg.dr.*` access → {doc}`user_guide/domain_randomization/index`

See the full {doc}`changelog`.

```

---

## Community & support

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🐛 Bugs & features
:link: https://github.com/unilabsim/UniLab/issues
GitHub Issues
:::

:::{grid-item-card} 💬 Discussions
:link: https://github.com/unilabsim/UniLab/discussions
Q&A, show-and-tell, RFCs
:::

:::{grid-item-card} 🤝 Contributing
:link: developer_guide/contributing
:link-type: doc
Set up dev env, run tests, file ADRs
:::

::::

## Citing UniLab

```bibtex
@misc{unilab2026,
  author = {UniLab Sim Authors},
  title  = {UniLab: A Universal Lab for Robot Learning},
  year   = {2026},
  url    = {https://github.com/unilabsim/UniLab}
}
```

```{toctree}
:hidden:
:caption: 🚀 Get Started

user_guide/getting_started/installation
user_guide/getting_started/quickstart
user_guide/getting_started/training
user_guide/getting_started/configuration_overrides
user_guide/getting_started/evaluation_and_playback
```

```{toctree}
:hidden:
:caption: 📚 User Guide

user_guide/index
user_guide/backends/index
user_guide/algorithms/overview
user_guide/tasks/g1_motion_tracking
user_guide/domain_randomization/index
user_guide/terrain/procedural
user_guide/manipulation/dexterous_inhand
user_guide/tooling/wandb_and_tensorboard
```

```{toctree}
:hidden:
:caption: 🔁 Transfer

transfer/index
transfer/sim_to_real/overview
transfer/sim_to_sim/why_switch
transfer/framework_migration/from_isaac_lab
```

```{toctree}
:hidden:
:caption: 🛠 Developer Guide

developer_guide/index
developer_guide/contributing
developer_guide/contributing_workflow
```

```{toctree}
:hidden:
:caption: 🇨🇳 中文文档 (zh_CN)

user_guide/zh_CN/01-getting-started
user_guide/zh_CN/02-simulation-backends
user_guide/zh_CN/03-training
user_guide/zh_CN/04-algorithms
user_guide/zh_CN/05-domain-randomization
developer_guide/zh_CN/development-standard
developer_guide/zh_CN/collaboration
developer_guide/zh_CN/CONTRIBUTING
developer_guide/zh_CN/domain-randomization-contract
developer_guide/zh_CN/scene-composition-design
developer_guide/zh_CN/motion-asset-migration
developer_guide/zh_CN/motrix-contact-sensor-notes
agents/zh_CN/01-agent-quick-reference
```

```{toctree}
:hidden:
:caption: 🧩 API Reference

api_reference/index
api_reference/base/index
api_reference/envs/index
api_reference/algos/index
api_reference/backend/index
api_reference/training/index
api_reference/dr/index
api_reference/top_level
```

```{toctree}
:hidden:
:caption: 📖 Reference

glossary
changelog
```
