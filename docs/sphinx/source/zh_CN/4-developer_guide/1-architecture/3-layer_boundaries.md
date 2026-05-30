# 分层边界

语言: 简体中文

本页是
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` 中所记录架构规则的
检查清单。项目的权威标准见
{doc}`/zh_CN/4-developer_guide/0-index`。

## Owner 层

| 层 | Owner 路径 | 拥有 |
| --- | --- | --- |
| L0 Backend | `src/unilab/base/backend/` | 物理 backend 抽象、由 backend 拥有的场景 materialization、backend 能力。 |
| L1 Env | `src/unilab/envs/`、`src/unilab/base/np_env.py` | MDP 语义、观测、奖励、reset 逻辑、backend 到任务的适配。 |
| L2 Config 与 Registry | `conf/`、`src/unilab/structured_configs.py`、`src/unilab/base/registry.py`、`src/unilab/training/reward.py` | Hydra compose、owner YAML 身份、env/reward 注册。 |
| L3 算法与 IPC | `src/unilab/algos/`、`src/unilab/ipc/` | Learner、runner、collector、replay 与 rollout buffer、权重同步。 |
| L4 Scripts | `scripts/` | 仅做入口装配。 |

## 规则

- 在 owner 层修复行为。训练脚本不应承载长期的 env、backend、reward 或算法业务
  规则。
- env 代码可以依赖 `src/unilab/base/backend/base.py` 中声明的 `SimBackend`
  契约；如果共享的 env 逻辑需要某个新的 backend 能力，先将其加入 `SimBackend`，
  再使用。
- Config 选择应保留在 `conf/` 下的 Hydra owner YAML 中，而不是放在 Python 层的
  backend 分支里。
- 资源、XML 与模型元数据相关的工作属于 init、materialization 或 cache 路径。
  不要把资源解析挪进 `step()`、`reset()` 或运行时 domain randomization 循环。

## 仓库中的证据

- 架构契约：{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries`
- Backend 边界：`src/unilab/base/backend/base.py`
- Env 状态契约：`src/unilab/base/np_env.py`
- Registry 构造路径：`src/unilab/base/registry.py`
- 训练入口：`scripts/train_rsl_rl.py`、`scripts/train_mlx_ppo.py`、
  `scripts/train_appo.py`、`scripts/train_offpolicy.py`

## Navigation

- Index: [文档](0-index.md)
