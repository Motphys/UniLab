# ADR-0001 Runtime Model And Layer Boundaries

- Status: Accepted
- Date: 2026-04-11
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

## Stable Contracts

- `registry.make(...)` 是 task 构造入口 contract。
- `NpEnvState.obs` 必须是 `dict`，`reset()` 返回 `(obs_dict, info_dict)`。
- `SimBackend` 是 backend 抽象边界；算法与脚本不应依赖后端私有实现。
- 异步路径统一复用 `AsyncRunner` 生命周期与 shared resource cleanup。

## Consequences

- 跨层问题必须在 owner layer 修复。
- 新功能必须先确定 contract 归属，再决定具体落点。
- 文档和评审用语应区分“稳定 contract”与“阶段性能力”。

## Alternatives Considered

- 保持脚本层按训练入口各自处理 backend/env/algo 差异。拒绝原因：会把 contract 漏洞扩散到 `scripts/`，并让 smoke run 掩盖 owner layer 问题。
- 只用顶层训练脚本定义 runtime contract。拒绝原因：无法约束 env、backend、runner 和 IPC 的长期边界。

## Evidence In Repo

- 架构基线文档: `docs/developers/zh_CN/development-standard.md`
- Backend 抽象: `src/unilab/base/backend/base.py`
- Env contract: `src/unilab/base/np_env.py`
- Registry 入口: `src/unilab/base/registry.py`
- Async runner: `src/unilab/ipc/async_runner.py`

## Related Documents

- [ADR Index](README.md)
- [RL Infrastructure 开发标准](../zh_CN/development-standard.md)
- [仿真后端](../../users/zh_CN/02-simulation-backends.md)
- [协作流程](../zh_CN/collaboration.md)
