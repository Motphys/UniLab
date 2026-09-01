---
orphan: true
---

# ADR-0001 Runtime Model And Layer Boundaries

- Status: Accepted
- Date: 2026-04-11
- Last updated: 2026-08-15
- Owners: Infra / Training maintainers
- Supersedes: None
- Superseded by: None

## Context

UniLab 同时支持多种算法入口和两种仿真后端。没有统一 runtime 与分层边界时，问题容易在 `scripts/` 临时修补，导致 contract 漏洞与跨层耦合。

## Decision

采用并坚持以下稳定架构边界:

1. Runtime 采用 `CPU Physics -> Collector/IPC -> GPU Learner` 的三段式零拷贝模型。
2. 分层依赖单向: `backend -> env -> config/registry -> algo/ipc -> scripts`。
3. `scripts/` 只做装配，不承载长期业务规则。
4. 变更评审优先检查 contract 是否被破坏，而不是只看 smoke run 是否通过。
5. 同步 PPO 默认单进程；单机多卡时按一卡一进程复制 env、policy 与 rollout，复用
   RSL-RL 的 startup model broadcast 和 per-mini-batch gradient averaging，不在
   UniLab 另建 PPO 同步协议。

## Stable Contracts

- `registry.make(...)` 是 task 构造入口 contract。
- `NpEnvState.obs` 必须是 `dict`，`reset()` 返回 `(obs_dict, info_dict)`。
- `SimBackend` 是 backend 抽象边界；算法与脚本不应依赖后端私有实现。
- 异步路径统一复用 `AsyncRunner` 生命周期与 shared resource cleanup。
- PPO 的 `algo.num_envs` 是 per-rank 数量；全局每轮新样本数为
  `world_size * num_envs * num_steps_per_env`。
- PPO 多卡只有 rank 0 持久化日志/checkpoint；rollout、GAE、normalizer buffer、
  curriculum 与 episode statistics 不做隐式全局聚合。

## Consequences

- 跨层问题必须在 owner layer 修复。
- 新功能必须先确定 contract 归属，再决定具体落点。
- 文档和评审用语应区分“稳定 contract”与“阶段性能力”。
- PPO launcher 只维护单机 device/rank 与 worker lifecycle；算法 collective 的版本
  兼容性继续由锁定的 RSL-RL 依赖负责。

## Alternatives Considered

- 保持脚本层按训练入口各自处理 backend/env/algo 差异。拒绝原因：会把 contract 漏洞扩散到 `scripts/`，并让 smoke run 掩盖 owner layer 问题。
- 只用顶层训练脚本定义 runtime contract。拒绝原因：无法约束 env、backend、runner 和 IPC 的长期边界。
- 要求用户手工调用 `torchrun`。拒绝原因：无法稳定提供 Hydra device list、共享运行
  目录、rank seed 与顶层 CLI 的失败传播语义。
- 在 UniLab 自行实现 PPO gradient collective。拒绝原因：会与 RSL-RL 已有的广播、
  adaptive-KL 和 mini-batch 同步周期形成第二套协议。

## Evidence In Repo

- 架构基线文档: `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md`
- Backend 抽象: `src/unilab/base/backend/base.py`
- Env contract: `src/unilab/base/np_env.py`
- Registry 入口: `src/unilab/base/registry.py`
- Async runner: `src/unilab/ipc/async_runner.py`
- PPO distributed adapter: `src/unilab/ipc/dp_launcher.py`,
  `src/unilab/training/rsl_rl.py`, `src/unilab/scripts/train_rsl_rl.py`
- PPO distributed tests: `tests/ipc/test_dp_launcher.py`,
  `tests/algos/test_rsl_rl_ppo.py`, `tests/scripts/test_train_script_configs.py`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
- {doc}`仿真后端 </zh_CN/2-user_guide/3-backends/0-index>`
- {doc}`协作流程 </zh_CN/4-developer_guide/5-contributing_workflow>`
