<h1 align="center"> UniLab </h1>

<h3 align="center">
Configure task semantics once. Run robot RL across physics backends.
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://github.com/unilabsim/UniLab/actions/workflows/ci.yml"><img src="https://github.com/unilabsim/UniLab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/paper-arXiv--2605.30313-red" alt="Paper"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="PyPI"></a>
  <a href="https://github.com/unilabsim/UniLab/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab Teaser" width="95%">
</p>

<p align="center"><em>One task-authoring surface for locomotion, manipulation, and motion tracking.</em></p>

UniLab is a complete, configurable product for robot reinforcement learning.
Describe a task with Hydra, assemble it from manager terms, select a physics
backend, and train or evaluate through one CLI. The same task-facing contract
connects CPU, GPU, and external-worker simulation to the learner runtime.

Physics adapters are provided by the independent
[`unisim-core`](https://github.com/unilabsim/unisim) package. RL algorithms and
their runners are provided by [`unilab-rl`](https://github.com/unilabsim/unilab_rl)
(Python namespace `uni_rl`). UniLab keeps the user-facing task, environment,
configuration, and experiment workflow together.

New to UniLab? Start with [First success](#first-success-run-a-demo). Already
have a task? Jump to [Train and evaluate](#train-and-evaluate) and change only
`--sim` to try another backend when a matching task owner is available.

## Highlights

```text
┌──────────────────────────────────────┐     Same task contract    ┌──────────────────────────────────────┐
│                                      │ ────────────────────────▶ │       Run it where you need          │
│       Define the task once           │                           │   MuJoCo · Motrix · MJWarp · Drake   │
│       Hydra · Managers · NumPy       │                           │    Genesis · IsaacGym · IsaacSim     │
│     Terms · rewards · commands       │                           │   CUDA · ROCm · macOS · MPS · XPU    │
│                                      │                           │         train · eval                 │
└──────────────────────────────────────┘                           └──────────────────────────────────────┘
```

UniLab's core idea is simple: define task semantics once as reusable
configuration, then change the simulator, hardware, or learner without
rewriting the task's environment lifecycle.

- **Configure, don't code.** Actions, observations, rewards, terminations,
  events, commands, curricula, and metrics are manager terms assembled in Hydra
  owner YAML. Variants built from existing terms need no new environment class
  — often no Python code at all.
- **Change the backend, keep the workflow.** Current and future simulators share
  the public `SimBackend` contract. Choose a backend with `--sim`; the same task
  authoring and train/eval workflow remains in place while the owner YAML keeps
  backend-specific details explicit.
- **Scale across the hardware you have.** CPU-parallel or external-worker
  simulation feeds accelerator learners through the injected env contract and
  async runtime. Algorithms and runners are supplied by the unified package
  ecosystem instead of being tied to one simulator.

## Getting started

The supported source workflow uses [`uv`](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# Fastest path to the first Motrix demo.
make setup-motrix

# Full local setup (MuJoCo + Motrix):
# make setup

# Optional platform/backend paths:
# make sync-rocm       # AMD GPU
# make sync-xpu        # Intel GPU
# make setup-drake     # Drake + native batch extension
```

The `mujoco` extra compiles a native extension and may require the platform
compiler and Python development headers. See the
[installation guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/2-installation.html)
for platform-specific setup, optional backends, and external worker runtimes.

## First success: run a demo

```bash
# Downloads the checkpoint and assets from Hugging Face on first run.
uv run demo dance
```

Available presets are `teaser`, `dance`, `wallflip`, `boxtracking`, `locomani`,
and `inhandgrasp`. Use `uv run demo --help` for device and refresh options.
The [quick demo guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/1-quick_demo.html)
explains rendering modes and server/macOS differences.

## Train and evaluate

```bash
# Train and replay a task with Motrix.
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Switch only the simulator for the same task.
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# Or use the same workflow with an off-policy learner.
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task g1_walk_flat --sim mujoco

# Headless video export.
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
  --load-run -1 --render-mode record
```

Route-defining choices are always visible:

```text
--algo + --task + --sim  →  Hydra owner YAML  →  registered environment
```

Use normal Hydra overrides after those flags:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

Do not override `training.sim_backend` to switch engines. It is the identity
field supplied by the selected owner YAML. Find resume, W&B, playback, and the
full command matrix in the
[training guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html).

## Manager-based configuration

UniLab wraps a community-familiar manager API with Hydra composition and a
NumPy runtime. A task owner can select and parameterize terms declaratively:

```yaml
env:
  observations:
    policy:
      terms:
        joint_pos:
          func: unilab.envs.mdp.joint_pos_rel
        command:
          func: unilab.envs.mdp.generated_commands
          params:
            command_name: twist
  actions:
    joint_pos:
      _target_: unilab.envs.mdp.JointPositionActionCfg
      entity_name: robot
      scale: 0.25
reward:
  tracking_lin_vel:
    func: unilab.tasks.locomotion.common.manager_terms.track_lin_vel_xy_exp
    weight: 1.0
```

This makes common task edits a config change: compose or disable a term, tune
its parameters, and reuse it across robots and backends without writing a new
environment class. The API follows the pinned mjlab manager semantics where
the contracts are shared, but it is a UniLab product with NumPy, Hydra, and
`NpEnvState` semantics. See the
[Manager-Based API guide](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/1-architecture/6-manager_based_api.html)
for the complete contract and known differences.

## Physics backends

Current backends are available through `unisim-core` and the same UniLab route;
the contract is designed to grow as new adapters land:

`mujoco` · `motrix` · `mjwarp` · `drake` · `genesis` · `isaacgym` · `isaacsim`

Choose one backend setup (combine extras when needed):

```bash
uv sync --extra mujoco
# uv sync --extra mujoco --extra motrix
# uv sync --extra mujoco --extra mjwarp
# uv sync --extra genesis
# make setup-drake
```

IsaacGym and IsaacSim use dedicated external worker environments. Backend
installation details, rendering behavior, and the evidence-based task support
matrix live in the
[backend guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/3-backends/0-index.html)
and [support matrix](https://unilabsim.github.io/UniLab-doc/en/5-reference/5-support_matrix.html).

## Ecosystem and assets

UniLab is designed to be the shared product surface for robot-specific
repositories. Current downstream examples include
[MicroDuck RL](https://github.com/unilabsim/microduck_rl_unilab) and
[EngineAI RL](https://github.com/unilabsim/engineai_rl_unilab). They can ship
robot recipes independently while consuming the same task, backend, and RL
contracts.

Large assets are fetched on cold paths instead of being bundled into the wheel:

- [Robot meshes and textures](https://huggingface.co/datasets/unilabsim/unilab-robots)
- [Motion clips](https://huggingface.co/datasets/unilabsim/unilab-motions)
- [Scenes](https://huggingface.co/datasets/unilabsim/unilab-scenes)
- [Grasp caches](https://huggingface.co/datasets/unilabsim/unilab-caches)
- [Demo checkpoints](https://huggingface.co/datasets/unilabsim/unilab-checkpoints)

For mainland China, set `HF_ENDPOINT=https://hf-mirror.com` when the default
Hugging Face endpoint is unreachable.

## Documentation

- [Documentation index](https://unilabsim.github.io/UniLab-doc/en/0-index.html)
- [Unified CLI reference](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/1-cli_reference.html)
- [Task and manager architecture](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/1-architecture/0-index.html)
- [Sim-to-sim deployment](https://unilabsim.github.io/UniLab-doc/en/3-deployment/2-sim_to_sim/1-backend_swap.html)
- [Algorithm extension recipe](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/3-extending/3-new_algorithm.html)
- [Architecture decisions](https://unilabsim.github.io/UniLab-doc/adr/ADR-0000-index.html)

For development and contribution workflows, see the
[contributing guide](CONTRIBUTING.md).

## Community

<p align="center">
  <img src="docs/sphinx/source/_static/assets/unilab-wechat-assistant.jpg" alt="UniLab community QR code" width="180">
</p>

<p align="center">Add the UniLab assistant on WeChat to join the community.</p>

## Citation

```bibtex
@article{jia2026unilab,
  title         = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author        = {Jia, Yufei and Cao, Zhanxiang and Yu, Mingrui and Zhang, Heng and Chen, Shenyu and Jiang, Dixuan and Li, Meng and Li, Xiaofan and Liu, Yiyang and Wu, Junzhe and Li, Zheng and Fang, XiLin and Cui, Tingyu and Fu, Shengcheng and Li, Haoyang and Wang, Anqi and Wang, Zifan and Zhu, Dongjie and Cao, Chenyu and Huang, Zhenbiao and Zheng, Ziang and Lu, Jie and Ma, Xin and Wei, Zhengyang and Zhao, Xiang and Zhan, Tianyue and He, Ye and Chen, Yuxiang and Jiang, Yizhou and Li, Yue and Ge, Haizhou and Dong, Yuhang and Jia, Fan and Zhang, Ziheng and Zhang, Meng and Deng, Xiwa and Chen, Zhixing and Shao, Hanyang and Dong, Chenxin and Li, Yixuan and Chen, Yizhi and Chen, Bokui and Zhang, Kaifeng and Cui, Hanqing and Qin, Yusen and Huang, Ruqi and Han, Lei and Wang, Tiancai and Li, Xiang and Gao, Yue and Zhou, Guyue},
  journal       = {arXiv preprint arXiv:2605.30313},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30313}
}
```

UniLab is released under the [Apache License 2.0](LICENSE). See the
independent [UniSim](https://github.com/unilabsim/unisim) and
[UniLab RL](https://github.com/unilabsim/unilab_rl) repositories for their
own release and citation information.

## Acknowledgments

UniLab would not exist without the excellent work of the
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) team and the
[mjlab](https://github.com/mujocolab/mjlab) developers and contributors. Isaac
Lab's manager-based API design and abstractions, together with mjlab's clear,
lightweight reference implementation, helped shape UniLab's Hydra and NumPy
task authoring experience. We sincerely thank both communities for sharing
their work and ideas.
