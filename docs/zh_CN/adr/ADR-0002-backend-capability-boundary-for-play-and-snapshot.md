# ADR-0002 Backend Capability Boundary For Play And Snapshot

- Status: Accepted
- Date: 2026-04-11
- Owners: Backend / Training maintainers

## Context

MuJoCo 与 Motrix 的渲染路径和输出形式不同。仓库当前存在两类事实:

1. 一部分路径依赖 MuJoCo 的 physics snapshot + video 导出。
2. 一部分路径依赖 Motrix 的交互式 renderer。

这属于 backend capability 差异，不是“同功能不同实现”的简单替换关系。

## Decision

将 play/render/snapshot 视为 capability contract，而不是统一行为 contract:

1. 允许 backend 在能力上不对等，不要求 feature parity。
2. 脚本层只依赖正式 capability 或 env-facing contract，不直接假设具体 backend 私有行为。
3. 文档和 support claim 明确区分:
   - 稳定 contract: 能力边界和调用约束。
   - backend-specific 差异: 输出形式、可用工具链、playback 体验。

## Stable Contracts

- 后端差异必须被显式建模为 capability，而不是散落在脚本判断里。
- 对外行为承诺是“是否支持某能力”，而不是“所有后端行为一致”。

## Backend-Specific Capability Differences

- MuJoCo: 具备 physics snapshot 驱动的视频导出路径。
- Motrix: 具备交互式 renderer 路径，常见 playback 体验是窗口渲染而非视频导出。

## Consequences

- 评审时不再以“是否与 MuJoCo 完全一致”作为 Motrix 路径验收标准。
- 新增 play/debug 逻辑必须先声明 capability 归属，再决定 fallback 或拒绝策略。

## Evidence In Repo

- 后端文档与矩阵: `docs/zh_CN/02-simulation-backends.md`
- Backend 抽象: `src/unilab/base/backend/base.py`
- 训练入口与 play 边界: `scripts/train_rsl_rl.py`, `scripts/train_mlx_ppo.py`, `scripts/train_appo.py`, `scripts/train_offpolicy.py`

## Related Documents

- [ADR Index](README.md)
- [RL Infrastructure 开发标准](../00-development-architecture.md)
- [仿真后端](../02-simulation-backends.md)
- [协作流程](../06-collaboration.md)
