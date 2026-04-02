# UniLab

UniLab 的核心目标是验证一个命题：**robot locomotion RL 不需要依赖 GPU 仿真后端**。

主流框架常把物理仿真、replay buffer 和策略训练全部耦合在 GPU pipeline 中。UniLab 选择另一条路线：**CPU 仿真 + shared-memory 数据通路 + GPU 训练**，在保持训练吞吐的同时降低对特定仿真后端和硬件平台的绑定。

```
┌───────────────────┐     统一共享内存     ┌────────────────────┐
│   CPU 物理仿真    │ ──────────────────▶  │   GPU 策略训练     │
│  mujoco.rollout   │   SharedReplayBuffer │  PPO / SAC / TD3   │
│    多线程并行     │  (PyTorch shared)    │    CUDA / MPS      │
└───────────────────┘                      └────────────────────┘
```

- **CPU 仿真**：MuJoCo / Motrix 的 CPU 多线程 step，无需 GPU 仿真内核
- **统一内存**：Collector 和 Learner 通过 PyTorch shared tensors 零拷贝通信
- **GPU 训练**：策略网络仍在 GPU 上训练，支持 CUDA 和 MPS
- **硬件无关**：Mac 和 Linux 都可以作为一等开发环境

## Quick Start

```bash
# 1. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 安装依赖
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA 12.4)
uv sync --extra dev --extra cu124

# 3. 运行一个训练任务
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

更完整的安装、后端和训练说明已经拆到按顺序组织的 `docs/`：

## Documentation

- [01 Getting Started](docs/01-getting-started.md): 安装、依赖、国内镜像、第一次运行
- [02 Simulation Backends](docs/02-simulation-backends.md): MuJoCo / Motrix 支持范围与使用方式
- [03 Training Guide](docs/03-training.md): 训练、回放、恢复训练、Hydra 参数、W&B
- [04 Algorithms](docs/04-algorithms.md): APPO、FastSAC、FastTD3 的用法与差异
- [05 G1 Motion Tracking](docs/05-g1-motion-tracking.md): G1 whole-body motion tracking 任务说明
- [06 Collaboration Workflow](docs/06-collaboration.md): GitHub issue / milestone / PR 协作方式
- [Contributing](CONTRIBUTING.md): 开发规范、测试、CI、提交流程

## Roadmap And Task Tracking

UniLab 不再把阶段任务、owner 和执行状态直接维护在 `README.md`。

- 用 GitHub Issues 跟踪 bug、feature、benchmark、docs 和 infra work
- 用 GitHub Milestones 跟踪阶段目标，例如 `M1`
- 用 [docs/06-collaboration.md](docs/06-collaboration.md) 统一说明协作规则

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
