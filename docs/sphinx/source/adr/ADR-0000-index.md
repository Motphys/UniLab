---
orphan: true
---

# Architecture Decision Records (ADR)

语言: 简体中文

本目录记录 UniLab 的稳定架构决策。这里不跟踪阶段性执行状态，只记录已经落地并作为评审基线的 contract。

新增 ADR 使用 [ADR Template](ADR-TEMPLATE.md)。ADR 只记录已经或即将作为 review 基线的结构性决策，不记录阶段性执行状态。

## ADR 列表

| ADR | Layer / Topic | Status |
|-----|---------------|--------|
| [ADR-0001 Runtime Model And Layer Boundaries](ADR-0001-runtime-model-and-layer-boundaries.md) | Runtime / layering | Accepted |
| [ADR-0002 Backend Capability Boundary For Play And Snapshot](ADR-0002-backend-capability-boundary-for-play-and-snapshot.md) | Backend capability | Accepted |
| [ADR-0003 Task Owner And Config Compose Contract](ADR-0003-task-owner-and-config-compose-contract.md) | Config owner | Accepted |
| [ADR-0004 Registry Bootstrap Contract](ADR-0004-registry-bootstrap-contract.md) | Registry bootstrap | Accepted |
| [ADR-0005 Unified Obs Critic Env And IPC Contract](ADR-0005-unified-obs-critic-env-and-ipc-contract.md) | Observation / IPC | Accepted |

## ADR Governance

每篇 ADR 应包含:

- `Status`
- `Date`
- `Owners`
- `Supersedes`
- `Superseded by`
- `Alternatives Considered`
- `Evidence In Repo`

当新 ADR 改写旧决策时，不直接删除旧 ADR；在旧 ADR 的 `Superseded by` 中指向新 ADR，并在新 ADR 的 `Supersedes` 中反向链接。

## 与主文档的关系

- 架构总览: {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
- 后端能力和支持矩阵: {doc}`仿真后端 </zh_CN/2-user_guide/3-backends/0-index>`
- 协作流程与证据标准: {doc}`协作流程 </zh_CN/4-developer_guide/5-contributing_workflow>`

## Navigation

- Index: {doc}`Documentation </index>`
