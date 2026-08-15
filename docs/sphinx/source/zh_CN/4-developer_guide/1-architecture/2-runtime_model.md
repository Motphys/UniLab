# 运行时模型

详细的运行时契约见
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` 与
{doc}`/zh_CN/4-developer_guide/0-index`。本页将运行时摘要与对应的代码路径放在
一起说明。

## 两种运行时形态

### 同步 PPO 路径

`scripts/train_rsl_rl.py` 会 compose Hydra config、
调用 registry bootstrap、通过 `registry.make(...)` 构造 env，并在同一进程内运行
learner。默认配置保持单进程；`training.devices` 指定多张卡时，父进程通过 PyTorch
elastic launcher 启动本机 worker，worker 再进入同一脚本完成上述构造。RSL-RL 路径
通过 `src/unilab/training/rsl_rl.py` 适配 `NpEnv`。

多卡时每个 rank 按配置创建完整的 `algo.num_envs`、policy copy 与 rollout storage，
数据和 GAE 不跨 rank 交换。RSL-RL 负责 startup model broadcast、adaptive-KL 标量
同步，以及每个 PPO mini-batch 的梯度平均。UniLab 的 launcher 只负责 device/rank、
共享 log dir、seed offset 与进程失败联动，不另建 PPO 同步协议。只有 rank 0 写日志与
checkpoint；normalizer buffer、curriculum 和 episode 统计保持 rank-local。

### 异步 APPO 与 off-policy 路径

APPO 与 off-policy runner 采用 CPU 仿真到 learner 的拆分：

```text
CPU physics env loop -> shared IPC buffer -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO 使用 `APPORunner`、`RolloutRingBuffer` 与 `SharedWeightSync`。
- SAC、TD3 与 FlashSAC 只使用一条 off-policy execution path：`ReplayBuffer`
  提供有界 host ingress，完整 ring 驻留在一个 CUDA/MPS learner device，
  `SharedWeightSync` 负责发布 actor 权重。
- `src/unilab/ipc/async_runner.py` 中的 `AsyncRunner` 负责 collector 进程启动、
  停止信号以及共享资源清理。

## 边界规则

- env 保持 numpy/向量化形态，并返回 `NpEnvState`。
- GPU tensor 与 optimizer 状态属于 learner 代码，而非 env 代码。
- Collector/learner 协议必须复用现有的 IPC 原语，而不是在 scripts 中另起临时的
  并行协议。
- PPO 多卡必须复用 RSL-RL 的 distributed contract；`algo.num_envs` 是 per-rank
  语义，不能在脚本中静默除以 world size。

## 仓库中的证据

- PPO 入口：`scripts/train_rsl_rl.py`
- APPO runner：`src/unilab/algos/torch/appo/runner.py`
- Off-policy runner：`src/unilab/algos/torch/offpolicy/double_buffer_runner.py`
- IPC 原语：`src/unilab/ipc/async_runner.py`、
  `src/unilab/ipc/rollout_ring_buffer.py`、`src/unilab/ipc/replay_buffer.py`、
  `src/unilab/ipc/weight_sync.py`
