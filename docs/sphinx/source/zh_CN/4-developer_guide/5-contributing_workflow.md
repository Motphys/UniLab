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

## AI Roadmap 与 Issue Scope 治理

本节约束由 AI agent 提出的 roadmap、architecture issue 和大型实施计划。技术完整性不能替代产品判断；测试、benchmark、review 和 gate 只能证明实现符合目标，不能证明目标值得做。

### Maintainer 能力与沟通基线

- UniLab maintainer 是仓库的主要开发者，具备较强的 RL、机器人仿真、Python、性能优化、配置与训练系统实践能力，也最了解 UniLab 的真实目标和历史约束。
- 不得把 maintainer 不熟悉 agent 自创术语、编译器式抽象、runtime 分层或企业化治理表述，解释为 maintainer “不专业”。如果主要开发者看不懂 roadmap，首先判定为 roadmap 表达失败。
- 默认使用中文和仓库中的现有名词。首次出现的新概念必须同时说明：它解决什么问题、会落在哪些现有模块、会新增什么长期负担。能用现有术语表达时不得发明 acronym、layer、runtime 或 protocol 名称。
- Maintainer 擅长根据具体代码、配置、数据流和性能事实做判断。先给具体影响，再给抽象模型；先说明“仓库会发生什么”，再说明“这个模式叫什么”。
- Agent 必须对 roadmap 的可理解性负责。不得用“已经有多个 AI review”“技术上自洽”或“所有 gate 可通过”代替 maintainer 本人的理解和判断。

### 先判断是否值得做

提出 roadmap 前，必须先用简短、直接的语言回答：

1. 这项工作直接服务 UniLab 的哪个核心目标？
2. 当前仓库中有什么证据说明问题真实存在？
3. 最小可行方案是什么？为什么不能只修 owner layer 或做一个薄 adapter？
4. 不做的具体代价是什么？
5. 会新增哪些需要长期维护的 contract、execution path、配置、测试或 CI？

其中任一项答不清，先调研或建议不做，不得直接生成大型 roadmap。Backend 适配案例不得自动升级为 production backend；性能探索不得自动升级为长期 support claim。

### Roadmap 的写法

Roadmap 必须分成两层，顺序不可颠倒：

1. **Owner summary**：使用普通中文，控制在约 300 字内，包含问题、推荐的最小方案、明确 non-goals、预估规模和需要 maintainer 决定的事项。
2. **Technical detail**：只在 owner summary 获得方向确认后展开，描述 owner boundary、数据流、风险和验证。详细类型、方法名、状态机或编译计划只能服务已确认的需求，不能用来制造方案已经成立的印象。

Owner summary 必须能让 maintainer 明确回答以下一句话：

> 这次只做什么，不做什么，完成后仓库会多出什么永久责任。

Roadmap 还必须遵守：

- 先给推荐方案，不先堆背景、术语和完整架构图。
- 需要选择时最多给 2–3 个真实选项，逐项写明用户价值、代码规模和长期成本；不能只比较实现优雅度。
- 每个新 abstraction 都要给一个仓库内的具体例子和一个“不采用它会怎样”的说明。
- 不提前设计尚未证明需要的 V2 interface、第二套 runtime/lifecycle、通用 compiler、完整 capability matrix 或 issue-specific evidence system。
- 多阶段 roadmap 只详细规划最近的 1–3 个可执行 issue；更远阶段只记录方向和启动条件，避免把猜测写成承诺。
- Roadmap 可作为 umbrella 记录方向，但 **umbrella 获批不等于实施获批**，也不授权 agent 自动依次执行所有 phase。

### 工作规模与 Issue 上限

每个 implementation issue 默认必须能由 maintainer 在一次 review 中理解，并由一个聚焦 PR 完成。

| 级别 | 默认规模 | 允许的内容 | 授权规则 |
|------|----------|------------|----------|
| Small | 约 1–8 个文件、≤ 400 行净手写改动 | 一个明确行为、一个 owner layer，以及贴近风险的测试/配置 | 可作为普通 issue 实施 |
| Standard | 约 9–15 个文件、≤ 800 行净手写改动 | 一个纵向切片，最多两个紧邻 owner layer，一个 PR | 实施前确认 scope summary |
| Large / Umbrella | 超过 15 个文件或 800 行；跨 2 个以上 owner layer；预计需要多个 PR | 只用于架构决策、拆分和依赖排序 | 不得直接实施，必须拆成 Small/Standard child issues |

以上数字是默认预算，不是鼓励用满的配额。预计或实际改动达到任一上限时必须暂停并重新评估。机械重命名、生成文件或批量配置迁移可以申请例外，但必须与行为变更分开，并单独报告手写代码和生成内容的规模。

一个 implementation issue 只能有一个主要结果。不得在同一 issue 中同时承担多个独立目标，例如：

- 公共 contract 重构 + 新 backend production 化；
- manager API 设计 + 两个 task 全量迁移 + 性能融合；
- correctness 实现 + 性能优化 + support 等级提升；
- feature 开发 + 永久 CI/benchmark/evidence 基础设施；
- 仓库清理 + 历史重写 + 新架构落地。

测试、文档和必要配置属于主结果的组成部分，不算第二个目标；可独立交付、可独立回退的能力必须另立 issue。

### Issue 的写法

Implementation issue 正文应保持短而可决策，主体建议不超过约 1,500 个中文字；更长的研究记录、接口草案或 benchmark 数据放入 ADR、文档或独立附件。Issue 必须按以下顺序书写：

1. **一句话问题**：当前具体哪里不对。
2. **为什么现在做**：引用现有代码、配置、测试、bug 或 benchmark 事实。
3. **最小交付结果**：合并后用户或仓库会得到什么。
4. **In scope**：3–7 条可审查的工作项。
5. **Non-goals**：明确列出容易顺手扩张但本次不做的内容。
6. **Owner 与预计改动**：owner layer、预计文件、手写 LOC 和 PR 数量。
7. **Acceptance criteria**：3–7 条靠现有 contract、局部测试或必要 benchmark 验证的结果。
8. **Stop conditions**：什么发现会导致暂停、拆 issue 或回到 maintainer 决策。

Issue 不得：

- 用数十个类型名、方法签名或 phase 掩盖尚未确认的产品选择；
- 把 speculative design 写成已经批准的 contract；
- 用 checklist 数量营造完成度；
- 为单个 issue 创建永久 claim inventory、freshness receipt、raw artifact、专属 CI gate 或项目管理框架；
- 把“以后可能独立成 package”“未来可能 production”写成本 issue 的隐含实施内容；
- 将多个 AI reviewer 的同意作为方案正确或 maintainer 已理解的证据。

### 确保 Maintainer 真正理解

在开始 Standard、Large、公共 contract 或跨层工作前，agent 必须完成一次理解确认：

1. 给出不依赖新术语的 owner summary；
2. 给出一条明确边界：`只做 X；不做 Y/Z`；
3. 用仓库路径或调用链说明改动落点；
4. 报告预估规模和合并后的永久维护项；
5. 让 maintainer 对真实 trade-off 做选择或明确确认该边界。

不要问空泛的“是否理解”。应询问会改变方案的具体问题，例如“只做统一 adapter，还是新增独立 execution path？”如果 maintainer 表示看不懂、无法复述目标，或对范围的理解与 roadmap 不一致，立即停止实施并用更短、更具体的语言重写。不得继续调用其他 AI review 来替代解释。

“写 roadmap”“新建 issue”或批准 umbrella，只授权规划和记录；不授权建分支、修改代码或执行所有 child issues。“开始开发”默认只授权当前明确确认的第一个 Small/Standard issue。完成一个 child issue 后，agent 不得自动进入下一个。

### 强制暂停条件

出现下列任一情况，agent 必须停止扩张，报告当前事实，并由 maintainer 重新决定继续、缩减、拆分或删除：

- 预计或实际规模超过 issue 中声明的文件/LOC/PR 预算；
- 需要新增公共 contract、runner、env lifecycle、training path 或同步协议；
- 一个 backend 的需求开始向 env、manager、runner 或 learner 扩散；
- 需要新增或修改常规 CI、support 等级、长期 benchmark 或 evidence infrastructure；
- 为了通过测试而需要引入 issue 原目标未提及的新 abstraction；
- 原定 adapter/case study 开始演变为 production subsystem；
- 实现过程中发现最小方案已经足够，或上游能力可以直接复用；
- Maintainer 无法清楚说明当前阶段完成后得到什么。

暂停不是失败。删除、缩减或拒绝一项技术上可实现但不服务 UniLab 核心目标的工作，是正确结果。

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
