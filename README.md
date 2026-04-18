![G1 motion tracking overview](docs/assets/g1_readme.png)

# UniLab

Languages: English | [简体中文](docs/users/zh_CN/01-getting-started.md)

Train robot RL without a GPU simulation backend.

UniLab uses **CPU simulation + shared-memory runtime + GPU learning** instead of coupling simulation and learning inside one GPU-resident pipeline.

Start with the `Quick Demo` below to run the first documented training command from this repository.

## Quick Demo

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
uv sync --extra motrix

# 3. Run a first PPO training job on macOS / MacBook
# macOS: (73s on M5Max-128GB, 1min43s on M3Max-48GB, 2.5min on MacBookNeo-8GB)
uv run mxpython scripts/train_rsl_rl.py task=go2_joystick_flat/motrix

# 4. Run the same training job on Linux
# Linux: PPO (31s on RTX 4090 and R9-9950x3d)
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
```

This is the shortest repository entrypoint today. It uses the PPO training script on the registered `go2_joystick_flat/motrix` task and gives a direct first-success path before you learn the full workflow.

On macOS / MacBook, commands that open the MotrixSim native renderer must be launched with `uv run mxpython` instead of `uv run python`. Plain non-rendering training can still use `uv run python ... training.no_play=true`.

## Example Runs

These are example repository runs for documented commands and hardware setups. They are useful as concrete entrypoints and reported timings, but they are **not** yet a formal benchmark manifest.

```bash
# Linux SAC (5.5min on RTX 4090 and R9-9950x3d)
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/motrix
```

```bash
# Linux G1 motion tracking (1h35min on RTX 4090 and R9-9950x3d)
uv run scripts/train_rsl_rl.py task=g1_motion_tracking/mujoco
```

## System Layout

```
┌───────────────────┐     Unified Shared Memory     ┌────────────────────┐
│  CPU Physics Sim  │ ───────────────────────────▶  │ GPU Policy Training│
│  mujoco.rollout   │      SharedReplayBuffer       │   PPO / SAC / TD3  │
│ Multithread Step  │    (PyTorch shared tensors)   │     CUDA / MPS     │
└───────────────────┘                               └────────────────────┘
```

## Workflow Entrypoints

| Goal | Entrypoint | Log root pattern |
|------|------------|------------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_ppo/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/rsl_rl_ppo/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` |
| TD3 | `scripts/train_offpolicy.py` | `logs/fast_td3/<task>/` |

Training scripts automatically enter playback after training unless you set `training.no_play=true`.

For MotrixSim visualization or `training.play_only=true` on macOS / MacBook, reuse the macOS renderer rule from `Quick Demo` and see `docs/users/zh_CN/03-training.md`.

## Repository Map

- `conf/`: Hydra configuration, including task / reward / algorithm composition
- `scripts/`: direct entrypoints for training, playback, motion preprocessing, and tooling
- `src/unilab/`: environments, backends, algorithms, and shared utilities
- `tests/`: unit tests, integration tests, and script configuration tests
- `docs/`: user documentation under `docs/users/` and developer documentation under `docs/developers/`

## Documentation

### For Users
- [01 Getting Started](docs/users/zh_CN/01-getting-started.md): installation, dependency setup, mirrors, and first-run commands
- [02 Simulation Backends](docs/users/zh_CN/02-simulation-backends.md): MuJoCo / Motrix support scope and backend selection
- [03 Training Guide](docs/users/zh_CN/03-training.md): training, playback, resume flow, Hydra overrides, and W&B
- [04 Algorithms](docs/users/zh_CN/04-algorithms.md): APPO, FastSAC, and FastTD3 usage and differences
- [05 G1 Motion Tracking](docs/users/zh_CN/05-motion-tracking.md): the G1 whole-body motion-tracking task
- [07 Domain Randomization](docs/users/zh_CN/07-domain-randomization.md): domain randomization configuration and best practices

### For Developers
- [CONTRIBUTING.md](CONTRIBUTING.md): development environment, commands, commit conventions, and PR workflow
- [RL Infrastructure Development Standard](docs/developers/zh_CN/development-standard.md): design principles, layering, contracts, and validation boundaries
- [Collaboration Workflow](docs/developers/zh_CN/collaboration.md): GitHub issue / milestone / PR collaboration rules and ADR governance
- [ADR Index](docs/developers/adr/README.md): architecture decision records

### For Agents
- [AGENTS.md](AGENTS.md): core principles, high-risk areas, and key file pointers

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
