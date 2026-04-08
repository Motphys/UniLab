![G1 motion tracking overview](../assets/g1_readme.png)

# UniLab

语言: [English](../../README.md) | 简体中文 | [日本語](../ja/README.md) | [한국어](../ko/README.md)

UniLab 的核心目标是验证一个命题: **robot locomotion RL 不需要依赖 GPU 仿真后端**。

主流框架常把物理仿真、replay buffer 和策略训练全部耦合在 GPU pipeline 中。UniLab 选择另一条路线: **CPU 仿真 + shared-memory 数据通路 + GPU 训练**。这样既能保持训练吞吐，也能降低对特定仿真器和硬件平台的绑定。

```text
┌───────────────────┐     统一共享内存通路      ┌────────────────────┐
│   CPU 物理仿真    │ ───────────────────────▶  │   GPU 策略训练     │
│  mujoco.rollout   │      SharedReplayBuffer   │   PPO / SAC / TD3  │
│    多线程并行     │   (PyTorch shared tensors)│     CUDA / MPS     │
└───────────────────┘                           └────────────────────┘
```

- **CPU 仿真**: MuJoCo / Motrix 的 CPU 多线程 step，不需要 GPU 仿真内核
- **统一内存通路**: collector 和 learner 通过 PyTorch shared tensors 零拷贝通信
- **GPU 训练**: 策略网络仍在 GPU 上训练，支持 CUDA 和 MPS
- **硬件无关**: macOS 和 Linux 都是一等开发环境

## Quick Start

```bash
# 1. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 安装依赖
# macOS (MPS)
uv sync

# Linux (任选一个 CUDA extra，例如 cu124)
uv sync --extra cu124

# 可选: Motrix 后端
uv sync --extra motrix

# 3. 运行一个训练任务
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

## Workflow Entrypoints

| 目标 | 入口脚本 | 默认日志根目录 |
|------|----------|----------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

训练脚本默认会在训练结束后自动进入回放；设置 `training.no_play=true` 可以跳过。

## Repository Map

- `conf/`: Hydra 配置，以及 task / reward / algorithm 组合
- `scripts/`: 训练、回放、motion 预处理和工具脚本直接入口
- `src/unilab/`: 环境、后端、算法和共享工具
- `tests/`: 单元测试、集成测试和脚本配置测试
- `docs/`: 四语文档，分别位于 `docs/en/`、`docs/zh_CN/`、`docs/ja/` 和 `docs/ko/`

## Documentation

- [00 RL Infrastructure Development Standard](00-development-architecture.md): 设计原则、分层结构、contract 和验证边界
- [01 Getting Started](01-getting-started.md): 安装、依赖同步、镜像和第一次运行
- [02 Simulation Backends](02-simulation-backends.md): MuJoCo / Motrix 支持范围与后端选择
- [03 Training Guide](03-training.md): 训练、回放、恢复训练、Hydra 覆盖参数和 W&B
- [04 Algorithms](04-algorithms.md): APPO、FastSAC 和 FastTD3 的用法与差异
- [05 G1 Motion Tracking](05-g1-motion-tracking.md): G1 全身 motion tracking 任务说明
- [06 Collaboration Workflow](06-collaboration.md): GitHub issue / milestone / PR 协作规则
- [Contributing](CONTRIBUTING.md): 开发流程、测试、CI 和 review 预期
- [AGENTS](../../AGENTS.md): coding agent 和自动化编辑器在这个 RL infra 仓库中的工作准则

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
