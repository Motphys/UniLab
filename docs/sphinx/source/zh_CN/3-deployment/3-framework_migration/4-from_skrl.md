# 从 skrl 迁移

skrl 的强项在于算法广度。UniLab 专注于一组精选算法（PPO、SAC、TD3，以及若干
优化变体），但增加了一条真实硬件部署路径。

## 把 skrl 概念映射到 UniLab

| skrl | UniLab |
|---|---|
| `Agent`（PPO、SAC……） | `unilab.algos.torch.*` |
| `RolloutMemory` | `unilab.ipc.rollout_ring_buffer` |
| `ReplayMemory` | `unilab.ipc.replay_buffer` |
| `Trainer` | `unilab.training.run` |
| 用于 env 的 `Wrapper` | 继承 `NpEnv` |

## 应当预期什么

- 对于小众算法（CQL、IQL 等）**没有算法对等** —— UniLab 有意专注于少数几个
  高度优化的 actor-critic 变体。
- **不同的 runner 生命周期。** skrl 那种单体式 trainer，变成了由共享内存连接的
  collector + learner 组合。参见
  {doc}`../../4-developer_guide/2-contracts/5-runner_lifecycle`。
- **不同的 env 接口。** skrl 容忍多种 env 风格。UniLab 坚持要求
  `NpEnv` + dict obs。

## 迁移清单

1. 决定哪个 UniLab 算法最匹配你的 skrl agent。
2. 把 env 移植为 `NpEnv` 形式。
3. 把超参数 YAML 转换为 `conf/<algo>/<task>/` 下的 Hydra 组。
4. 验证 reward 一致性。
