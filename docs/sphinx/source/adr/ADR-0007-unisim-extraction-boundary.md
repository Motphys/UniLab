---
orphan: true
---

# ADR-0007 UniSim Extraction Boundary

语言: 简体中文

- Status: Accepted
- Date: 2026-09-02
- Owners: Backend / Packaging / Env / Config maintainers
- Supersedes: None
- Superseded by: None

## Context

UniLab 已在 `unisim.backend` 形成统一 `SimBackend` contract，并注册
MuJoCo、Motrix、Drake、MJWarp、Genesis、IsaacGym 和 IsaacSim。该层仍依赖 UniLab 的
scene、asset、domain-randomization、dtype、playback 与 runtime helper，无法被物理引擎
benchmark 或其他消费者独立使用。

Roadmap #1428 将统一物理层拆到 GitHub 仓库 `unilabsim/unisim`。PyPI distribution 使用
`unisim-core`，Python import namespace 使用 `unisim`。本 ADR 固定 owner boundary 和迁移
约束；不把 manager/task/training runtime 搬入 core，也不建立 benchmark 专属 backend API。

## Decision

### Package and repository

- `unisim-core` 是 distribution 名称，`unisim` 是唯一 public Python namespace。
- 新仓库为 public `unilabsim/unisim`，沿用 Apache-2.0。
- 迁移优先保留可追溯历史（filtered-history/import）；不重写 UniLab 现有分支历史。
- 版本沿用 UniLab 当前版本策略；每个发布版本记录 source commit、迁移说明和兼容范围。

### Ownership

`unisim-core` owns：

- backend-neutral state/control/reset/mutation 类型、capability、错误和生命周期；
- lazy adapter factory/registry、engine-native runtime 与必要的 subprocess IPC；
- cold-path materialization/selector 和 hot-path buffer/step 规则；
- conformance helper、benchmark API/result schema 预留、package 文档与 TestPyPI 发布。

UniLab owns：

- Hydra owner YAML、task/env/manager lifecycle、`NpEnvState`、reward/observation/termination；
- runner、learner、checkpoint、sim2sim、robot asset registry、task XML/scene composition；
- 将 task-owned scene、DR、dtype/config 翻译为 UniSim 输入的 adapter layer。

`unisim` 不得 import UniLab、Hydra、Torch、Gymnasium、RSL-RL、训练脚本或 task code。
UniLab env/manager 不得访问 engine model/data/private runtime。

### Migration and final state

首发纵向切片为 core + MuJoCo + Motrix + conformance；benchmark 只预留 API、result schema
和 provenance 字段，当前不实现 workload、测量或结论。

roadmap 最终必须迁移并验证全部七类 backend：MuJoCo、Motrix、Drake、MJWarp、Genesis、
IsaacGym、IsaacSim。迁移完成后，UniLab 删除对应实现、重复测试和 `unilab.base.backend`
compatibility shim，不保留长期双实现。

每个 adapter child 必须同时提交代码、focused conformance、support/optional-extra 文档
和中英文迁移说明。out-of-process adapter 共用一份 subprocess protocol。

### Branch and release governance

- UniLab declared base 是 `dev/issue-1042-manager-based-api`，执行期间只读；`main` 同样只读。
- roadmap 集成分支为 `dev/issue-1428-unisim-extraction`，从 declared base 的只读快照创建。
- child branch/PR 只合入 roadmap 集成分支；不得向 UniLab `main` 或 declared base 写入。
- 开发与验证阶段只发布到 TestPyPI，凭据来自 `~/.pypirc` 且不得输出或提交；生产 PyPI
  发布在 roadmap 完成后由 maintainer 手动执行。

## Alternatives Considered

- 继续把 backend 留在 UniLab：无法为独立 benchmark 提供轻量 consumer，依赖边界继续扩大。
- distribution 与 import 均使用 `unisim-core`：不符合已确认的 public import API。
- 一次迁移全部 adapter 后再发布首版：风险集中且无法尽早验证 package boundary；采用分阶段
  release，但不缩小最终全量迁移范围。

## Evidence In Repo

- `unisim.backend.base`：统一 `SimBackend` contract。
- `src/unilab/base/backend_factory.py`：backend factory 与 lazy adapter loading。
- `tests/base/test_backend_conformance.py`、`tests/base/test_backend_imports.py`：现有边界测试。
- #888 / PR #892：第一阶段可提取 physics boundary。
- #1428：跨仓拆分 roadmap、child 顺序与验收标准。

## Related Documents

- `docs/sphinx/source/zh_CN/4-developer_guide/2-contracts/2-backend_contract.md`
- `docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md`
- `docs/sphinx/source/zh_CN/4-developer_guide/1-architecture/3-layer_boundaries.md`
