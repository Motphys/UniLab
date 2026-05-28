# UniLab Glossary

语言: 简体中文

本页定义 UniLab 文档中反复出现的核心术语。架构和评审文档应链接到这里，避免同一个概念在多处重新解释。

## Backend Capability

后端显式声明并实现的能力边界。MuJoCo 和 Motrix 不要求功能完全对等；play、snapshot、video、domain randomization 等差异必须通过 capability 或 owner config 表达。

## Cold Path / Hot Path

Cold path 指 init、materialization、cache build 等低频路径。Hot path 指 `step()`、`reset()`、domain randomization 执行循环等高频路径。

资产、XML、模型元数据只能在 cold path 访问。hot path 不允许解析 asset/XML，也不允许通过 backend 私有属性探测来决定运行时行为。

## Evidence Grades

用于描述 backend/task/algo 支持程度的证据等级：

- `Registered`: registry 中存在 env/backend 注册。
- `Configured`: 存在对应 owner YAML。
- `Tested`: `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合。
- `Benchmarked`: 存在绑定该组合的 benchmark manifest。
- `Recommended`: 存在显式 recommendation metadata。

没有对应证据时，不应把能力写成稳定支持或推荐路径。

## Owner Layer

拥有某类规则和默认值的最低合理层。问题应在 owner layer 修复，而不是在上层脚本绕过 contract。

常见归属：

- backend 差异归 backend adapter 或 backend-specific config。
- MDP、observation、reward、reset 归 env。
- task/backend/reward/algo 组合归 owner YAML。
- collector/learner 生命周期归 runner / IPC。
- 脚本只做流程装配。

## Owner YAML / Task Owner

`task=<task>/<backend>` 指向的最终 task owner 配置。PPO / APPO 路径位于 `conf/{ppo,appo}/task/<task>/<backend>.yaml`，off-policy 路径位于 `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`。

owner YAML 直接持有 `training.task_name`、`training.sim_backend`、`reward`、`env` 以及 task-specific `algo`。`training.sim_backend` 是 owner YAML 的身份字段，不是独立 backend switch。

## Registry Bootstrap

通过 `unilab.base.registry.ensure_registries()` 导入显式声明的 registry modules，使 `@registry.envcfg(...)` 和 `@registry.env(...)` decorator 生效。bootstrap 入口是 package contract，不依赖目录扫描推断注册目标。

## Task Owner Config

同 Owner YAML。文档中出现 task owner config 时，强调它是 task/backend/reward/algo 组合的唯一入口，而不是可被 `training.sim_backend=...` 单独改写的拆分式配置。

## Related Documents

- {doc}`仿真后端 </zh_CN/user_guide/02-simulation-backends>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/developer_guide/development-standard>`
- {doc}`ADR-0003 Task Owner And Config Compose Contract </adr/ADR-0003-task-owner-and-config-compose-contract>`
