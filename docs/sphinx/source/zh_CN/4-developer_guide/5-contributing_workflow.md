# 协作工作流

仓库文档只记录稳定的标准。执行状态、负责人与阶段推进应当存放在 GitHub
协作对象中。

如果你只是想安装或训练 UniLab，请从
{doc}`/zh_CN/1-getting_started/2-installation` 与
{doc}`/zh_CN/1-getting_started/1-quick_demo` 开始。

## 工作项粒度

每个 issue 至少应当回答以下问题：

1. 我们要解决什么问题？
2. 期望的交付物是什么？
3. 完成标准是什么？
4. 谁负责执行？
5. 存在哪些上游阻塞？

推荐的 issue 类型：

- `bug`
- `work item`：feature / infra / benchmark / test / sim / docs 类工作

## Milestone 结构

每个 milestone 应当：

- 作为 GitHub 中的 milestone 对象存在
- 拥有一个聚合各 sub-issue 的 tracking issue
- 把执行细节放在 sub-issue 中，而不是 milestone 描述里
- 以交付的产物定义完成，而不只是“代码已合并”

典型的完成产物：

- 绿色 CI
- benchmark 结果或 W&B run 链接
- demo 视频 / ONNX 导出 / checkpoint 路径
- 如果用户可见行为发生变化，需附带文档更新

## PR 证据标准

每个 PR 应当：

- 关联驱动该工作的 issue
- 描述用户可见的改动与训练影响
- 列出实际执行过的验证命令
- 说明行为在 `mujoco`、`motrix`、macOS 或 Linux 之间是否变化

## 所有权模型

执行 owner 通过 GitHub assignee 表达，review owner 通过 `CODEOWNERS` 表达。
如果尚无稳定的 GitHub handle，可暂时不指派该 issue，并在 issue 正文中临时
注明预期的 owner。

## ADR 治理

当一项改动涉及 runtime / backend / config / registry 契约时，该 issue 或 PR
必须显式关联对应的 ADR：

- 架构标准入口：{doc}`Architecture Overview </zh_CN/4-developer_guide/1-architecture/1-overview>`
- ADR 索引：{doc}`ADR Index </adr/ADR-0000-index>`
- 后端能力边界：{doc}`ADR-0002 </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- 任务 owner / compose：{doc}`ADR-0003 </adr/ADR-0003-task-owner-and-config-compose-contract>`
- Registry bootstrap：{doc}`ADR-0004 </adr/ADR-0004-registry-bootstrap-contract>`

如果现有 ADR 无法覆盖某项新的结构性决策，请在同一个 PR 中新增一份 ADR，
并将其反向链接进上述文档。
新的 ADR 使用 {doc}`ADR Template </adr/ADR-TEMPLATE>`，且必须显式说明
`Supersedes`、`Superseded by`、`Alternatives Considered` 与 `Evidence In Repo`。
