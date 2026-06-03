<h1 align="center"> UniLab </h1>

<h3 align="center">
A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/arxiv-2605.30313-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/"><img src="https://img.shields.io/badge/paper-UniLab-orange" alt="Paper"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="https://discord.gg/EPCuguRmGX"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab Teaser" width="95%">
</p>

<p align="center"><em>Train robot RL without a GPU simulation backend. Teaser rendered with MotrixSim.</em></p>

Start with the `Quick Demo` below to run the primary training command. The recommended setup uses `uv`; Conda and pip users should still follow the `uv` workflow for now. Platform-specific notes and current boundaries are in the [installation guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/2-installation.html).

## ✨ Highlights

```
┌───────────────────┐                            ┌─────────────────────────┐
│  CPU Physics Sim  │   Unified Shared Memory    │   GPU Policy Training   │
│   MuJoCo/Motrix   │ ─────────────────────────▶ │     PPO / SAC / TD3     │
│ Multithread Step  │    SharedReplayBuffer      │ CUDA / MPS / ROCm / XPU │
└───────────────────┘                            └─────────────────────────┘
```

- **Heterogeneous RL runtime:** CPU-parallel simulation streams transitions through shared memory while policy learning runs on GPU accelerators.
- **Two physics backends:** MuJoCoUni and MotrixSim are integrated through backend-specific adapters and task owner configs.
- **Unified training CLI:** `uv run train` and `uv run eval` cover PPO, MLX PPO, APPO, SAC, TD3, and FlashSAC; additional HORA and HIM-PPO paths are documented as script-level workflows.
- **Config-owned tasks:** Hydra owner YAML files select task, reward, backend, and algorithm settings together; backend switching is expressed as `task=<task>/<backend>`.
- **Cross-platform setup paths:** The repository tracks Linux CUDA, Linux ROCm, Linux XPU, and Apple Silicon / macOS setup flows.

## 🚀 Quick Demo

<table>
  <tr>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/dance.jpg" alt="dance demo" width="100%">
      <br>
      <sub><b>dance</b><br>G1 motion tracking</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/wallflip.jpg" alt="wallflip demo" width="100%">
      <br>
      <sub><b>wallflip</b><br>G1 wall flip</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/teaser.jpg" alt="teaser demo" width="100%">
      <br>
      <sub><b>teaser</b><br>MotrixSim teaser</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/boxtracking.jpg" alt="boxtracking demo" width="100%">
      <br>
      <sub><b>boxtracking</b><br>G1 box tracking</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/inhandgrasp.jpg" alt="inhandgrasp demo" width="100%">
      <br>
      <sub><b>inhandgrasp</b><br>Sharpa in-hand</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/locomani.jpg" alt="locomani demo" width="100%">
      <br>
      <sub><b>locomani</b><br>Go2 loco-manipulation</sub>
    </td>
  </tr>
</table>

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
# Pick the setup command for your platform.

# Linux CUDA or macOS
make setup-motrix
# Without shell completion setup: uv sync --extra motrix
# If `make` is not installed: uv sync --extra motrix && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu

# 3. Pre-trained checkpoint playback (downloads from Hugging Face on first run)
uv run demo dance
```

Available demo names: `teaser`, `dance`, `wallflip`, `boxtracking`, `locomani`, `inhandgrasp`. See the [Unified CLI](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/1-cli_reference.html) page for the full list and flags.

> Mainland China users: motions, scenes, and demo checkpoints are pulled from Hugging Face on first run. If `huggingface.co` is unreachable, point the client at the community mirror before running demo commands:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

For training and evaluation:

```bash
uv run train --algo appo --task go2_joystick_flat --sim motrix

uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1

# Headless Motrix video export for Linux/server runs
uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

This routes through the `go2_joystick_flat/motrix` task owner config and keeps backend selection explicit.

On macOS / MacBook, the UniLab CLI routes Motrix interactive playback through `mxpython` when needed. Motrix defaults to interactive playback; use `--render-mode record` for headless video export or `--render-mode none` to skip playback. Detailed script-level commands are in the [Training Guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html).

## 🏃 Example Runs

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

```bash
uv run train --algo sac --task g1_motion_tracking --sim motrix
```

```bash
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora
```

> Grasp caches auto-download from Hugging Face (`unilabsim/unilab-caches`) on first run into `src/unilab/assets/caches/`; no manual step is needed. To regenerate locally for custom scales (slow):
>
> ```bash
> bash scripts/sharpa_collect_grasps.sh 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
> ```

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim motrix
uv run eval --algo ppo --task go2_arm_manip_loco --sim motrix --load-run -1
```

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
```

Use `uv run train` for training, `uv run eval` for checkpoint playback, and `uv run demo` for the local demo preset. These commands keep algorithm, task, and backend selection explicit.

More training commands, script-level entrypoints, algorithm matrix, resume flow, and W&B details are in the [Training Guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html).

## 📚 Documentation

Use the published [UniLab documentation](https://unilabsim.github.io/UniLab-doc/); start at the [English documentation index](https://unilabsim.github.io/UniLab-doc/en/0-index.html). High-signal entrypoints:

- [Getting Started](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/0-index.html): installation, Docker runtime, dependency setup, and first-run commands
- [Training Guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html): training, playback, resume flow, Hydra overrides, and W&B
- [Simulation Backends](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/3-backends/0-index.html): generated MuJoCo / Motrix support matrix
- [Development Standard](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/0-index.html): contracts, layering, and validation boundaries
- [ADR Index](https://unilabsim.github.io/UniLab-doc/adr/ADR-0000-index.html): accepted architecture decisions

## 💬 Community

Join our [Discord server](https://discord.gg/EPCuguRmGX) to chat with the community and get help.

| WeChat Group | WeChat Assistant |
| :---: | :---: |
| <img src="docs/sphinx/source/_static/assets/unilab-wechat-group.jpg" alt="UniLab WeChat group QR code" width="220"> | <img src="docs/sphinx/source/_static/assets/unilab-wechat-assistant.jpg" alt="UniLab WeChat assistant QR code" width="150"> |
| Scan to join the UniLab WeChat group. | If the group is full, add the assistant and include `unilab交流` in your message. |

## 🧾 Citation

### UniLab

```bibtex
@article{jia2026unilab,
  title         = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author        = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal       = {arXiv preprint arXiv:2605.30313},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30313}
}
```

### Physics Backends

```bibtex
@article{jia2026mujocouni,
  title  = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year   = {2026}
}

@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}
```
