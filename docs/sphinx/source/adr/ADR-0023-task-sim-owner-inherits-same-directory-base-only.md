---
orphan: true
---

# ADR-0023 Task Sim Owner Inherits Same-Directory Base Only

语言: 简体中文

- Status: Proposed
- Date: 2026-11-18
- Owners: Config / Env maintainers
- Supersedes: None
- Superseded by: None

## Context

`src/unilab/conf/<algo>/task/<task>/` 下的仿真 owner YAML（backend leaf）历史上
出现了跨后端继承：`g1_motion_tracking` 的 motrix/mjwarp/genesis/newton leaf 直接
继承同目录的 `mujoco.yaml`，`go2_joystick_flat/superdex.yaml` 同样继承
`mujoco.yaml`。这带来两个问题：

- 隐式耦合：leaf 的组合结果依赖另一个后端的 owner，修改 MuJoCo owner 的
  reward/env/algo 会静默改变 Motrix/MJWarp/Genesis 的训练行为，review 无法从
  leaf 文件本身看出其真实契约。
- 身份混淆：mujoco owner 同时承担"任务 canonical contract"与"MuJoCo 后端身份"
  两个角色，违背 ADR-0003 的 task owner 身份语义。

2026-11-18 的清理确认了该问题的范围：sac/flashsac/appo/ppo 四个算法树的
`g1_motion_tracking` 与 ppo `go2_joystick_flat` 共 10 个 leaf 违规，其中
appo/ppo 的 `g1_motion_tracking` 目录连 `base.yaml` 都不存在，mujoco owner
被迫充当 base。

## Decision

- 每个 `<algo>/task/<task>/` 目录必须包含一个 `base.yaml`，持有该任务对该算法的
  canonical contract：scene/observations/actions/commands/terminations/reward，
  canonical DR `env.events`，以及后端共享的 training budget（`training.task_name`、
  `algo` 预算）。
- 目录内每个仿真 backend leaf（`mujoco.yaml`、`mjwarp.yaml`、`genesis.yaml`、
  `motrix.yaml`、`newton.yaml`、`superdex.yaml`、`drake.yaml` 等）的 Hydra
  `defaults:` 只允许列出 `/task/<task>/base` 与 `_self_`，禁止继承任何其他
  backend leaf。leaf 直接持有身份字段（`training.task_name`、
  `training.sim_backend`，见 ADR-0003）、渲染/设备 placement，以及显式支持
  或禁用的 DR term（如 `events.push_robot: null`）。
- 若某后端确有独立的调参需求（如 superdex 的 `max_iterations`、观测归一化），
  必须在该 leaf 内显式写出完整数值，不允许通过继承其他后端 owner 获得。
- `base.yaml` 自身不是可训练入口（不持有 `training.sim_backend`），仅作为
  contract 声明与 leaf 的 compose 基线。

## Stable Contracts

- `src/unilab/conf/<algo>/task/<task>/<backend>.yaml` 的 `defaults:` 恰好包含
  `/task/<task>/base` 与 `_self_`；该结构由
  `tests/config/test_config_system.py` 的继承结构测试全量守护。
- backend leaf 的组合结果必须能从 `base.yaml + leaf 文件本身` 完整推出；
  review 时 leaf 文件即其完整 backend 契约。
- 后端间共享的 DR/奖励定义收敛在 `base.yaml`；单个后端不支持某 term 时，
  在该后端 leaf 中显式置 `null` 或收窄参数，禁止"借道"其他后端 owner。
- 跨算法对齐（如 flashsac 与 sac 的 `g1_motion_tracking` 任务契约）通过使两边
  `base.yaml` 一致来表达，而不是通过跨树继承。

## Alternatives Considered

- **保留跨后端继承、仅加注释约束：** 拒绝。继承链使行为变更不可见，正是本次
  清理要消除的问题；注释无法被测试守护。
- **允许继承同任务任意 leaf、只禁止跨任务继承：** 拒绝。同任务跨后端继承同样
  让 MuJoCo owner 的修改静默传播到其他后端，违背"leaf 文件即完整契约"目标。
- **用 Hydra config group（`task/g1_motion_tracking` 组默认值）代替显式
  defaults：** 拒绝。组默认值是隐式路由，review 时同样看不到 leaf 的 compose
  基线；显式 `defaults: [/task/<task>/base, _self_]` 一目了然。

## Consequences

- 后端 owner 之间出现少量显式数值重复（如 superdex 携带观测归一化与学习率）。
  这是有意的：重复可见优于继承隐式。
- 修改 `base.yaml` 等同于修改该任务所有后端的契约，必须按 ADR-0003 的
  owner 变更流程评审；修改单个 leaf 不再影响其他后端。
- 新增 backend leaf 时必须先确认同目录 `base.yaml` 存在；目录内没有
  `base.yaml` 时禁止新增 leaf。
- 2026-11-18 的迁移以"组合结果不变"为验收标准：除
  flashsac `g1_motion_tracking` 按本仓库决策向 sac 契约对齐（补齐 canonical DR、
  `geom_names`、actor 有偏关节观测）外，所有受影响 owner 的 Hydra 组合输出
  与迁移前逐键一致。

## Evidence In Repo

- owner 配置树：`src/unilab/conf/{sac,flashsac,appo,ppo}/task/<task>/`。
- 对齐示例：flashsac `g1_motion_tracking` 的 `base.yaml` 与 sac 同任务
  `base.yaml` 契约一致（scene/observations/commands/reward/terminations/DR）。
- 继承结构守护测试：`tests/config/test_config_system.py`。
- 组合等价与任务契约测试：`tests/config/test_g1_motion_tracking_owners.py`。

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0003 Task Owner And Config Compose Contract </adr/ADR-0003-task-owner-and-config-compose-contract>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
