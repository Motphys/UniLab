<h1 align="center"> UniLab </h1>

<h3 align="center">
Contract-driven infrastructure for robot learning across physics backends and hardware
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://github.com/unilabsim/UniLab/actions/workflows/ci.yml"><img src="https://github.com/unilabsim/UniLab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/paper-arXiv--2605.30313-red" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/CoRL-2026-orange" alt="CoRL 2026"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="PyPI"></a>
  <a href="https://github.com/unilabsim/UniLab/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<h3 align="center">🎉 🎉 UniLab has been accepted to <b>CoRL 2026</b>! 🎉 🎉</h3>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab Teaser" width="95%">
</p>

<p align="center"><em>One task-authoring surface for locomotion, manipulation, and motion tracking.</em></p>

UniLab is configurable infrastructure for robot reinforcement learning.
Describe a task with Hydra, assemble it from manager terms, select a physics
backend, and train or evaluate through one CLI. The same task-facing contract
connects CPU, GPU, and external-worker simulation to the learner runtime.

The same framework has documented paths for Windows, Apple Silicon macOS, Linux
CUDA, AMD ROCm, and Intel XPU. Backend and task maturity are evidence-graded;
use the [support matrix](https://unilabsim.github.io/UniLab-doc/en/5-reference/5-support_matrix.html)
to choose a tested combination.

See policies in action on the [project page](https://unilabsim.github.io/#demos),
or read [Why UniLab?](https://unilabsim.github.io/UniLab-doc/en/why_unilab.html)
to understand the project fit, evidence, and comparison with alternatives.

## Highlights

UniLab's core idea is simple: define task semantics once as reusable
configuration, then change the simulator, hardware, or learner without
rewriting the task's environment lifecycle.

- **Configure, don't code.** Actions, observations, rewards, terminations,
  events, commands, curricula, and metrics are manager terms assembled in Hydra
  owner YAML. Variants built from existing terms need no new environment class
  — often no Python code at all.
- **Change the backend, keep the workflow.** Registered simulators meet the
  public `SimBackend` contract. Choose a backend with `--sim`; when a matching
  task owner exists, task authoring and train/eval stay consistent while
  backend-specific details remain explicit.
- **Keep solver and learner devices independent.** CPU-parallel, native, or
  external-worker simulation can feed an accelerator learner without first
  becoming a CUDA-resident simulator. The learner can run on CUDA, ROCm, MPS,
  or XPU; the [support matrix](https://unilabsim.github.io/UniLab-doc/en/5-reference/5-support_matrix.html)
  records the evidence level of each backend/task combination.
- **Accelerate replay-based off-policy training.** FastSAC/FlashSAC lets
  simulation data collection overlap with learner updates. The paper reports
  3–10× end-to-end gains on representative configurations; see [Why UniLab](https://unilabsim.github.io/UniLab-doc/en/why_unilab.html)
  for scope and measurements.

## Quick start

The supported source workflow uses [`uv`](https://docs.astral.sh/uv/). This is
the shortest path to a policy demo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

make setup
# Downloads the checkpoint and assets from Hugging Face on first run.
uv run demo dance
```

For Windows, macOS, CUDA, ROCm, XPU, optional backends, and headless rendering,
use the [installation guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/2-installation.html)
and [quick demo guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/1-quick_demo.html).

## Train and evaluate

```bash
# Train and replay one task with Motrix.
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Use the same task-facing command with another configured backend.
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# Replay-based off-policy path.
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

The flags keep algorithm, task, and simulator choices visible. Resume, W&B,
Hydra overrides, playback, backend setup, and the full command matrix belong in
the [training guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html),
[backend guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/3-backends/0-index.html),
and [support matrix](https://unilabsim.github.io/UniLab-doc/en/5-reference/5-support_matrix.html).

## Ecosystem

UniLab is designed to be a shared task and training surface for robot-specific
repositories. They can ship robot recipes independently while consuming the
same task, backend, and RL contracts. Current downstream examples:

- [MicroDuck RL](https://github.com/unilabsim/microduck_rl_unilab)
- [EngineAI RL](https://github.com/unilabsim/engineai_rl_unilab)
- [Wuji](https://github.com/unilabsim/wuji_unilab)
- [Legged Manipulation](https://github.com/unilabsim/legged-manipulation_unilab)

## Documentation

- [Why UniLab?](https://unilabsim.github.io/UniLab-doc/en/why_unilab.html)
- [Installation and first demo](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/0-index.html)
- [Training and evaluation](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html)
- [Backend support matrix](https://unilabsim.github.io/UniLab-doc/en/5-reference/5-support_matrix.html)
- [Sim-to-sim deployment](https://unilabsim.github.io/UniLab-doc/en/3-deployment/2-sim_to_sim/1-backend_swap.html)
- [Developer guide](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/0-index.html)

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
