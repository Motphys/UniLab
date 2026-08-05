---
orphan: true
---

# ADR-0006 Structured Term Plan And NumPy Tier 1 Boundary

- Status: Accepted
- Date: 2026-08-05
- Owners: Env / Performance maintainers
- Supersedes: None
- Superseded by: None

## Context

G1 walk 与 motion tracking 的 NumPy table 和手写 Numba kernel 需要最小的 resolved contract，且不能改变 `NpEnv` lifecycle。

## Decision

1. `unilab.term` 只依赖标准库和 NumPy，拥有 plan/Tier 1，不拥有 task lifecycle。
2. definition 声明 key、tensor/参数和 callable；config 顺序即 canonical order。
3. cold path 拒绝重复/未知 term、非法参数和冲突 spec；materialization 一次绑定
   shape/dtype、分配 buffer 并预建 context。
4. scale 是 runtime 值；零值跳过 callable，其他数值输出原地乘 scale，termination 只允许 `0/1`。
5. Numba item 参数顺序固定；后续 Tier 2 必须自动对拍。

## Stable Contracts

- warm `execute()` 只访问 resolved terms/数组；task owner 继续负责 combine 和 lifecycle。
- 缺失 Numba item 在 Tier 1 合法；Tier 2 必须显式 fail closed，不能静默混跑。

## Alternatives Considered

- manager/compiler 或 DSL：拒绝，复杂度与双 pilot 不成比例。
- Python 逐 env 调用 item：拒绝，会丢失 vectorized authoring。
- 继续手工同步 table/order/kernel：拒绝，无法形成 resolved plan。

## Consequences

只交付 synthetic Tier 1；真实 task/Numba assembler 分属后续 issue。

## Evidence In Repo

- `src/unilab/term/`、`tests/term/`；GitHub issues `#918`、`#919`

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
