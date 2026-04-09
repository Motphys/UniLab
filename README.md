![G1 motion tracking overview](docs/assets/g1_readme.png)

# UniLab

Languages: English | [简体中文](docs/zh_CN/README.md) | [日本語](docs/ja/README.md) | [한국어](docs/ko/README.md)

UniLab is built to test a simple hypothesis: **robot locomotion RL does not need a GPU simulation backend**.

Mainstream stacks often couple physics simulation, replay storage, and policy updates inside one GPU pipeline. UniLab takes a different route: **CPU simulation + shared-memory data path + GPU training**. That keeps training throughput high while reducing coupling to a specific simulator or hardware platform.

```
┌───────────────────┐     Unified Shared Memory     ┌────────────────────┐
│  CPU Physics Sim  │ ───────────────────────────▶  │ GPU Policy Training│
│  mujoco.rollout   │      SharedReplayBuffer       │   PPO / SAC / TD3  │
│ Multithread Step  │    (PyTorch shared tensors)   │     CUDA / MPS     │
└───────────────────┘                               └────────────────────┘
```

- **CPU simulation**: MuJoCo / Motrix CPU multithreaded stepping, no GPU sim kernel required
- **Unified memory path**: collectors and learners communicate through zero-copy PyTorch shared tensors
- **GPU training**: policy networks still train on GPU, with CUDA and MPS support
- **Hardware agnostic**: macOS and Linux are both first-class development environments

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
# macOS (MPS)
uv sync

# Linux (choose one CUDA extra such as cu124)
uv sync --extra cu124

# Optional: Motrix backend
uv sync --extra motrix

# 3. Run a training job
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

## Workflow Entrypoints

| Goal | Entrypoint | Default log root |
|------|------------|------------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

Training scripts automatically enter playback after training unless you set `training.no_play=true`.

## Repository Map

- `conf/`: Hydra configuration, including task / reward / algorithm composition
- `scripts/`: direct entrypoints for training, playback, motion preprocessing, and tooling
- `src/unilab/`: environments, backends, algorithms, and shared utilities
- `tests/`: unit tests, integration tests, and script configuration tests
- `docs/`: language-specific documentation under `docs/en/`, `docs/zh_CN/`, `docs/ja/`, and `docs/ko/`

## Documentation

- [00 RL Infrastructure Development Standard](docs/en/00-development-architecture.md): design principles, layering, contracts, and validation boundaries
- [01 Getting Started](docs/en/01-getting-started.md): installation, dependency setup, mirrors, and first-run commands
- [02 Simulation Backends](docs/en/02-simulation-backends.md): MuJoCo / Motrix support scope and backend selection
- [03 Training Guide](docs/en/03-training.md): training, playback, resume flow, Hydra overrides, and W&B
- [04 Algorithms](docs/en/04-algorithms.md): APPO, FastSAC, and FastTD3 usage and differences
- [05 G1 Motion Tracking](docs/en/05-g1-motion-tracking.md): the G1 whole-body motion-tracking task
- [06 Collaboration Workflow](docs/en/06-collaboration.md): GitHub issue / milestone / PR collaboration rules
- [Contributing](CONTRIBUTING.md): development workflow, testing, CI, and review expectations
- [AGENTS](AGENTS.md): guidance for coding agents and automated editors working in this RL infra repo

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
