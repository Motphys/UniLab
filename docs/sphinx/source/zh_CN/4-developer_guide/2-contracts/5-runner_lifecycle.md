# Runner 生命周期

语言: 简体中文

Runner 代码拥有训练生命周期。脚本负责 compose Hydra config 并启动正确的
runner；它们不应另起第二套 collector/learner 协议。

## 共享入口流程

各训练脚本遵循相同的高层序列：

1. 从算法 config 根目录与所选任务 owner YAML compose 出 Hydra config。
2. 调用 `ensure_registries()`。
3. 通过 `registry.make(...)` 或共享的 `create_env(...)` helper 构造 env。
4. 构建算法 runner 或 trainer。
5. 训练、checkpoint，按需 play back，并关闭自身拥有的资源。

## 各运行时的 owner

- `scripts/train_rsl_rl.py` 使用 `RslRlVecEnvWrapper` 与 RSL-RL 的
  `OnPolicyRunner`。
- `scripts/train_mlx_ppo.py` 使用 MLX PPO 的 trainer 路径。
- `scripts/train_appo.py` 使用 `APPORunner`、`RolloutRingBuffer` 与
  `SharedWeightSync`。
- `scripts/train_offpolicy.py` 使用 off-policy runner，配合 `ReplayBuffer` 与
  `SharedWeightSync`。
- `AsyncRunner` 为异步 runner 拥有 collector 进程生命周期与共享资源清理。

## 规则

- 不要为异步 collector 绕开 `AsyncRunner.close()` 的语义。
- 不要在 runner 代码内部修补 env 观测或 critic 语义；保持 `obs` 加可选 `critic`
  的契约。
- 使用 `src/unilab/training/run.py` 中共享的 log-root、checkpoint 与 playback
  解析 helper，而不要把这些规则复制到脚本中。

## 仓库中的证据

- 共享训练 helper：`src/unilab/training/common.py`、
  `src/unilab/training/run.py`
- 异步生命周期：`src/unilab/ipc/async_runner.py`
- Runner 测试：`tests/algos/test_appo_runner.py`、
  `tests/algos/test_offpolicy_runner.py`、`tests/ipc/test_async_runner.py`

## Navigation

- Index: [文档](0-index.md)
