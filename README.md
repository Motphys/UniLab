![G1 motion tracking overview](docs/assets/g1_readme.png)

# UniLab

Languages: English | [简体中文](docs/zh_CN/01-getting-started.md)

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
# 0. If uv not installed:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
# macOS (MPS, installs PyPI torch wheels)
# Linux (default: installs PyTorch cu128 wheels)
# Requires an NVIDIA GPU and driver stack supported by current PyTorch cu128 wheels
uv sync --extra motrix

# 3. Run a training job
# macOS (73s on M5Max-128GB, 1min43s on M3Max-48GB, 2.5min on MacBookNeo-8GB)
uv run mxpython scripts/train_rsl_rl.py task=go2_joystick/motrix

# Linux (31s on RTX 4090 and R9-9950x3d)
uv run scripts/train_rsl_rl.py task=go2_joystick/motrix

# Linux (5.5min on RTX 4090 and R9-9950x3d)
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_sac/motrix

# Linux (1h35min on RTX 4090 and R9-9950x3d)
uv run scripts/train_rsl_rl.py task=g1_motion_tracking/mujoco
```

On macOS / MacBook, commands that open the MotrixSim native renderer must be launched with `uv run mxpython` instead of `uv run python`. Plain non-rendering training can still use `uv run python ... training.no_play=true`.

## Workflow Entrypoints

| Goal | Entrypoint | Log root pattern |
|------|------------|------------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/<algo.algo_log_name>/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/<algo.algo_log_name>/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/<algo.algo_log_name>/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/<algo.algo_log_name>/<task>/` |

The concrete log directory comes from `algo.algo_log_name`. Current defaults are `rsl_rl_ppo`, `appo`, `fast_sac`, and `fast_td3`.

Training scripts automatically enter playback after training unless you set `training.no_play=true`.

For MotrixSim visualization on macOS / MacBook, use `uv run mxpython`:

```bash
uv run mxpython scripts/train_rsl_rl.py task=go1_joystick/motrix training.play_only=true
```

## Repository Map

- `conf/`: Hydra configuration, including task / reward / algorithm composition
- `scripts/`: direct entrypoints for training, playback, motion preprocessing, and tooling
- `src/unilab/`: environments, backends, algorithms, and shared utilities
- `tests/`: unit tests, integration tests, and script configuration tests
- `docs/`: language-specific documentation under `docs/zh_CN/`

## Documentation

- [00 RL Infrastructure Development Standard](docs/zh_CN/00-development-architecture.md): design principles, layering, contracts, and validation boundaries
- [01 Getting Started](docs/zh_CN/01-getting-started.md): installation, dependency setup, mirrors, and first-run commands
- [02 Simulation Backends](docs/zh_CN/02-simulation-backends.md): MuJoCo / Motrix support scope and backend selection
- [03 Training Guide](docs/zh_CN/03-training.md): training, playback, resume flow, Hydra overrides, and W&B
- [04 Algorithms](docs/zh_CN/04-algorithms.md): APPO, FastSAC, and FastTD3 usage and differences
- [05 G1 Motion Tracking](docs/zh_CN/05-g1-motion-tracking.md): the G1 whole-body motion-tracking task
- [06 Collaboration Workflow](docs/zh_CN/06-collaboration.md): GitHub issue / milestone / PR collaboration rules
- [07 Domain Randomization](docs/zh_CN/07-domain-randomization.md): domain randomization configuration and best practices
- [Contributing](CONTRIBUTING.md): development workflow, testing, CI, and review expectations
- [AGENTS](AGENTS.md): guidance for coding agents and automated editors working in this RL infra repo

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
