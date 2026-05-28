# ADR-0003 Task Owner And Config Compose Contract

- Status: Accepted
- Date: 2026-04-11
- Owners: Config / Env maintainers
- Supersedes: None
- Superseded by: None

## Context

历史上 task、backend、reward、algo 组合可能分散在多个位置，导致:

- 用户通过 `training.sim_backend=...` 误以为可以独立切后端。
- 脚本层补丁掩盖 owner YAML 的真正身份语义。

## Decision

确立 `task owner YAML` 为后端与任务组合的唯一入口 contract:

1. 组合入口是 `task=<task>/<backend>`（offpolicy 还包含 algo 维度）。
2. owner YAML 直接持有 `training.task_name`、`training.sim_backend`、`reward`、`env` 及 task-specific `algo`。
3. `training.sim_backend` 是 owner 身份字段，不是独立 backend switch。
4. CLI override 允许参数覆盖，但不能破坏 task owner 的 backend identity。

## Stable Contracts

- PPO/APPO owner 路径: `conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Offpolicy owner 路径: `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`
- reward 注入与 backend 差异表达必须在 owner YAML 层显式存在。

## Consequences

- 文档示例和 issue 模板必须使用完整 `task=.../<backend>` 形式。
- 配置行为变更应先改 owner YAML，再评估是否需要代码改动。

## Alternatives Considered

- 保留 task、backend、reward、algo 的拆分式配置组。拒绝原因：用户可以通过局部 override 组合出没有 owner 的状态，破坏 backend identity。
- 允许 `training.sim_backend=...` 单独切换后端。拒绝原因：它会绕过 owner YAML 中的 reward/env/backend-specific algo 配置。

## Evidence In Repo

- 架构标准与配置语义: `docs/sphinx/source/zh_CN/developer_guide/development-standard.md`
- 后端选择说明: `docs/sphinx/source/zh_CN/user_guide/02-simulation-backends.md`
- 配置测试: `tests/config/test_config_system.py`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/developer_guide/development-standard>`
- {doc}`仿真后端 </zh_CN/user_guide/02-simulation-backends>`
- {doc}`协作流程 </zh_CN/developer_guide/collaboration>`
