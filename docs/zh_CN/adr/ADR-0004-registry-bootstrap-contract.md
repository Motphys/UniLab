# ADR-0004 Registry Bootstrap Contract

- Status: Accepted
- Date: 2026-04-11
- Owners: Infra / Registry maintainers

## Context

训练入口和工具脚本依赖 registry 中的 env 注册结果。若 bootstrap 规则不清晰，常见问题是:

- 依赖隐式 import side effect，行为难以推断。
- 文档只能描述“先跑某脚本再说”，缺少明确 contract。

## Decision

将 registry bootstrap 作为独立 contract 记录，并要求调用方显式遵守:

1. registry 以 `registry.make(...)` 作为实例化入口。
2. bootstrap 过程由统一入口触发，保障 env decorators 已执行。
3. support claim 与矩阵生成都以 bootstrap 后的 registry 事实为准。

## Stable Contracts

- 构造 contract: `src/unilab/base/registry.py`
- bootstrap 入口: `unilab.utils.algo_utils.ensure_registries()` 及其 training helper 包装
- support claim 证据路径: registry 列表、owner YAML、测试清单

## Consequences

- 新增 env/task 时，需要同时更新注册入口和 owner YAML。
- 文档和工具不得把“扫描目录是否恰好导入成功”当成 contract 本身。

## Evidence In Repo

- Registry 实现: `src/unilab/base/registry.py`
- Bootstrap 使用面: `src/unilab/training/common.py`, `scripts/train_*.py`
- 支持矩阵生成说明: `docs/zh_CN/02-simulation-backends.md`

## Related Documents

- [ADR Index](README.md)
- [RL Infrastructure 开发标准](../00-development-architecture.md)
- [仿真后端](../02-simulation-backends.md)
- [协作流程](../06-collaboration.md)
