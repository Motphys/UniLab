# 运行时模型

语言: 简体中文

详细的运行时契约见
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` 与
{doc}`/zh_CN/4-developer_guide/0-index`。本页将运行时摘要与对应的代码路径放在
一起说明。

## 两种运行时形态

### 同步 PPO 路径

`scripts/train_rsl_rl.py` 与 `scripts/train_mlx_ppo.py` 会 compose Hydra config、
调用 registry bootstrap、通过 `registry.make(...)` 构造 env，并在同一进程内运行
learner。RSL-RL 路径通过 `src/unilab/training/rsl_rl.py` 适配 `NpEnv`；MLX 路径
则使用 `src/unilab/algos/mlx/ppo/runner.py` 与 `src/unilab/algos/mlx/ppo/ppo.py`。

### 异步 APPO 与 off-policy 路径

APPO 与 off-policy runner 采用 CPU 仿真到 learner 的拆分：

```text
CPU physics env loop -> shared IPC buffer -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO 使用 `APPORunner`、`RolloutRingBuffer` 与 `SharedWeightSync`。
- SAC、TD3 与 FlashSAC 使用 off-policy runner，配合 `ReplayBuffer` 与
  `SharedWeightSync`。
- `src/unilab/ipc/async_runner.py` 中的 `AsyncRunner` 负责 collector 进程启动、
  停止信号以及共享资源清理。

## 边界规则

- env 保持 numpy/向量化形态，并返回 `NpEnvState`。
- GPU tensor 与 optimizer 状态属于 learner 代码，而非 env 代码。
- Collector/learner 协议必须复用现有的 IPC 原语，而不是在 scripts 中另起临时的
  并行协议。

## 仓库中的证据

- PPO 入口：`scripts/train_rsl_rl.py`、`scripts/train_mlx_ppo.py`
- APPO runner：`src/unilab/algos/torch/appo/runner.py`
- Off-policy runner：`src/unilab/algos/torch/offpolicy/runner.py`
- IPC 原语：`src/unilab/ipc/async_runner.py`、
  `src/unilab/ipc/rollout_ring_buffer.py`、`src/unilab/ipc/replay_buffer.py`、
  `src/unilab/ipc/weight_sync.py`

## Navigation

- Index: [文档](0-index.md)
