# ADR-0004 Registry Bootstrap Contract

- Status: Accepted
- Date: 2026-04-11
- Owners: Env / Infra maintainers
- Supersedes: None
- Superseded by: None

## Context

UniLab 的 env 注册依赖 `@registry.envcfg(...)` 与 `@registry.env(...)` decorator side effect。若 bootstrap 继续依赖目录扫描和隐式导入，问题会有三个:

1. registry contract 由文件系统布局隐式决定，而不是由 package 明确声明。
2. 导入失败边界不清晰，review 时很难区分“缺失 optional package”与“bootstrap contract 破坏”。
3. support claim 与架构文档无法稳定引用 bootstrap 入口。

## Decision

将 env registry bootstrap 定义为显式 package contract:

1. registry bootstrap 入口仍然是 `ensure_registries()`。
2. package 若承担 bootstrap 责任，必须显式声明其 registry module 列表。
3. `ensure_registries()` 只导入声明的 bootstrap modules，不再依赖目录扫描推断注册目标。
4. optional package 与 required package 的失败策略继续分离: optional 记录 warning，required 直接失败。

## Stable Contracts

- `ensure_registries()` 是 registry-based entrypoint 的统一 bootstrap 入口。
- package-level bootstrap 清单必须是显式声明，而不是通过扫描推断。
- bootstrap contract 破坏必须在导入阶段直接暴露。

## Consequences

- 新增 env package 时，需要同步声明 bootstrap modules。
- registry 相关回归可以在 `ensure_registries()` 边界直接测试，不必依赖顶层训练脚本间接发现。
- 文档可以把 registry bootstrap 作为正式架构引用，而不是“当前实现细节”。

## Alternatives Considered

- 继续通过目录扫描推断 registry modules。拒绝原因：注册 contract 会被文件系统布局隐式决定，导入失败边界不清晰。
- 只在训练脚本里按需导入 env package。拒绝原因：support claim 和 config compose 无法稳定复用同一个 bootstrap contract。

## Evidence In Repo

- Registry 入口: `src/unilab/base/registry.py`
- Bootstrap helper: `src/unilab/base/registry.py`
- Env package 入口: `src/unilab/envs/locomotion/__init__.py`, `src/unilab/envs/motion_tracking/__init__.py`, `src/unilab/envs/manipulation/__init__.py`
- Bootstrap tests: `tests/utils/test_algo_utils.py`, `tests/base/test_registry.py`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/developer_guide/development-standard>`
- {doc}`协作流程 </zh_CN/developer_guide/collaboration>`
