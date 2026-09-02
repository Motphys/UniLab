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

本节延续 [discussion #883](https://github.com/unilabsim/UniLab/discussions/883)
对项目方向、知情决策和长期维护责任的关注，并把实施流程整理为清晰决策、合理
粒度、集成分支和按 base 分流的 CI。它适用于 AI agent 提出的 roadmap、architecture
issue 和多 PR 实施计划，目标是在规范开发的同时保持连续交付效率。

### 可理解的沟通基线

- UniLab maintainer 负责产品方向和长期维护选择，并依据具体代码、配置、数据流与
  性能事实作出判断。
- Roadmap 作者负责把方案翻译成仓库已有概念。默认使用中文和仓库现有名词，先说明
  仓库会发生什么，再补充必要的抽象名称。
- 新概念首次出现时，同时说明它解决的问题、对应的现有模块和新增的长期责任，并给出
  一个仓库内实例。
- AI review、测试、benchmark 与 gate 提供实现证据；owner summary 与 maintainer
  的明确选择记录产品判断。
- 当 maintainer 需要更多说明时，作者用更短的表述、具体路径和真实选项重新说明，直到
  双方对交付边界形成一致理解。

### 价值与最小方案

Roadmap 首先用简短、直接的语言回答：

1. 这项工作服务 UniLab 的哪个核心目标？
2. 哪些现有代码、配置、测试、bug 或 benchmark 证明当前机会真实存在？
3. 最小且完整的方案是什么，现有 owner layer 或上游能力可以复用到什么程度？
4. 预期收益、机会成本和优先级依据是什么？
5. 合并后会新增哪些 contract、execution path、配置、测试、CI 或 support 责任？

证据仍在形成时，先把工作定义为 research、benchmark 或 adapter case study，并把升级为
production/support 的条件写成后续决策点。这样每个支持等级都有与之匹配的仓库证据。

### Roadmap 的内容结构

Roadmap 按以下两层组织：

1. **Owner summary**：使用普通中文，约 300 字，包含目标、推荐方案、交付边界、
   预估规模、永久维护责任和需要 maintainer 决定的事项。
2. **Technical detail**：在方向确认后展开 owner boundary、数据流、依赖、风险、验证、
   child issue 和集成顺序。类型、方法名、状态机与性能计划都对应已确认的交付需求。

Owner summary 应让 maintainer 能够清楚复述：

> 这次交付什么，范围边界在哪里，完成后仓库会长期维护什么。

Roadmap 的具体写法遵循以下原则：

- 推荐方案位于背景、术语和架构细节之前。
- 需要选择时提供 2–3 个真实选项，并逐项说明用户价值、代码规模和长期成本。
- 每个新 abstraction 配一个仓库内实例，同时说明采用现有结构的可行路径与取舍。
- 近期 child issue 写到可执行和可验收；远期内容记录方向、依赖与启动条件，并随证据更新。
- Roadmap 记录一个集成结果、declared base branch、child issue 列表、依赖顺序和最终
  验收方式。开发授权后再记录对应集成分支。

### 以交付边界决定 Issue 粒度

Issue 粒度服务于理解、review 和交付效率：

| 类型 | 主要用途 | 交付方式 |
|------|----------|----------|
| Roadmap | 定义一个需要多个 PR 共同完成的集成结果、关键决策和验收边界 | 通过集成分支汇总 child PR，最终合回 roadmap declared base |
| Implementation | 交付一个可观察、可审查的主要结果，以及完成该结果所需的代码、配置、测试和文档 | 通常由一个聚焦 PR 完成 |
| Research / Benchmark | 形成可复现证据和明确决策结论 | 产物可独立审查，并为后续 implementation 提供输入 |

Implementation issue 采用完整纵向切片。以下内容适合保留在同一个 issue 中：

- 同一行为所需的 contract、owner 实现、配置、测试和用户文档；
- 为保持端到端可运行而需要同步调整的相邻 owner layer；
- 只有组合完成才具备验收价值的迁移步骤；
- 主结果验收所需的 benchmark、兼容性处理和清理工作。

当子项具备以下任一特征时，拆成 child issue 能提升协作效率：

- 可以独立产生用户或仓库价值，并有自己的验收标准；
- 具有独立的架构选择、风险确认、review owner 或交付节奏；
- 可以独立回退，且通过稳定接口与其他子项协作；
- 并行开发可以明显缩短周期，同时依赖边界已经清楚。

文件数、手写 LOC、目录数、owner layer 数和 PR 数作为 review 工作量与排期的规划信号。
Issue 始终按独立交付价值划分；预估发生明显变化时，更新规模、依赖和 review 方案，再
选择最连贯的交付边界。一个 helper、一份配置、一组测试或一段配套文档通常与其服务的
主要结果放在一起。

### Issue 的写法

Implementation issue 正文保持简短、具体并可直接执行。复杂研究记录、接口草案和大段
benchmark 数据可放入 ADR、文档或附件。正文按需要包含：

1. **问题与证据**：当前机会或缺口，以及对应仓库事实。
2. **主要交付结果**：合并后用户或仓库获得的完整能力。
3. **范围边界**：本 issue 包含的工作，以及关联但独立交付的工作。
4. **Owner 与 contract 影响**：主要 owner layer、相邻边界和长期责任。
5. **Roadmap 关系与 target branch**：parent roadmap、roadmap declared base、依赖
   issue 和 PR base。
6. **规模与 review 计划**：预计文件、手写 LOC、PR 组织方式和适合的 reviewer。
7. **Acceptance criteria 与 validation**：可观察结果、局部测试和必要 benchmark。
8. **范围复核点**：哪些新发现需要更新方案或请 maintainer 选择。

产品选择使用 owner summary 显式呈现；技术细节服务于已确认的边界；长期 CI、evidence
或 support 设施对应可复用的长期需求；未来设想以启动条件记录。AI review 结论作为参考
证据附在 maintainer 决策之后。

### Roadmap 集成分支工作流

Roadmap 获得明确开发授权后，按以下流程推进：

1. 在 roadmap issue 中记录 declared base branch。它可以是 `main`，也可以是上层
   roadmap 的集成分支；从该 base 的最新 head 创建并推送
   `dev/issue-<roadmap-number>-<slug>`。该命名延续仓库现有 `dev/issue-*` 惯例。
2. 每个 child issue 从最新集成分支创建工作分支。分支前缀表达改动类型，例如
   `feat/issue-<number>-<slug>`、`fix/issue-<number>-<slug>`、
   `refactor/issue-<number>-<slug>`、`perf/issue-<number>-<slug>`、
   `test/issue-<number>-<slug>` 或 `docs/issue-<number>-<slug>`。
3. Child branch 在最终 review head 与最新集成分支对齐，运行贴近风险的测试和本地
   `make test-all`，并把实际命令与结果写入 PR。
4. Child PR 的 base 设置为本 roadmap 集成分支。本地 gate 与 review 通过后合入；这类
   PR 使用本地验证结果，远程执行留给实际 base 为 `main` 的 PR。
5. 已批准 roadmap 范围内的 child issues 可以按依赖顺序连续推进；roadmap 中声明的产品
   checkpoint 仍在对应位置完成确认。
6. Child issues 全部集成后，在集成分支最新 head 再运行 `make test-all`，随后创建从
   `dev/issue-...` 合回 declared base 的最终 PR。
7. 最终 PR 按实际 base 选择 gate：base 为 `main` 时等待当前 head 的远程 CI；base 为
   其他分支时采用本地 gate 与 review，并由后续进入 `main` 的 PR 承担远程验证。合入完成
   后按仓库维护习惯清理本 roadmap 的集成分支和 child branches。

当 declared base 在 roadmap 开发期间前进时，在计划好的集成点同步该 base，并在同步后的
head 重新执行本地 gate。并行 child issues 在进入 review 前同步当前集成分支，使每个
本地结果都覆盖实际合入候选。

### 授权与范围复核

授权分为两个清晰阶段：

- **规划授权**：编写 roadmap、创建 issue、研究证据和整理 child issue；仓库实现保持在
  当前状态。
- **开发授权**：普通 implementation issue 获得该 issue 的实现权限；roadmap issue 获得
  已确认交付边界和 child issue 集合的连续实现权限，并启用对应集成分支工作流。

以下长期责任作为 roadmap 的显式决策项：公共 contract、execution path、runner/env
lifecycle、training path、同步协议、常规 CI、support 等级、长期 benchmark/evidence
设施、历史重写，以及 adapter 向 production subsystem 的升级。开发授权覆盖 owner
summary 中已经确认的决策项；实施期间新增的决策项先更新 owner summary、影响范围与长期
成本，再由 maintainer 确认。

以下情况触发一次范围复核：

- 实际规模、依赖或永久维护责任相对 issue 预估出现明显变化；
- backend 改动延伸到原范围外的 env、manager、runner 或 learner contract；
- 测试结果表明需要新的 abstraction 或长期基础设施；
- 上游复用或更小方案已经可以满足主要结果；
- maintainer 需要更具体的路径、调用链或 trade-off 说明。

范围复核用当前事实、推荐选项和影响说明支持 maintainer 选择继续、调整、拆分或结束该项
工作。选择记录回 roadmap 或 implementation issue，后续开发从更新后的边界继续。

## Milestone 结构

每个 milestone 应当：

- 作为 GitHub 中的 milestone 对象存在
- 拥有一个聚合各 sub-issue 的 tracking issue
- 把执行细节放在 sub-issue 中，而不是 milestone 描述里
- 以交付的产物定义完成，而不只是“代码已合并”

典型的完成产物：

- 与 PR base 对应的验证记录：本地 `make test-all`，以及实际 base 为 `main` 时的绿色
  远程 CI
- benchmark 结果或 W&B run 链接
- demo 视频 / ONNX 导出 / checkpoint 路径
- 如果用户可见行为发生变化，需附带文档更新

## PR 证据标准

每个 PR 应当：

- 关联驱动该工作的 issue
- 记录 base branch；child PR 同时关联 parent roadmap 与集成分支
- 描述用户可见的改动与训练影响
- 列出实际执行过的验证命令，并记录最终本地 head 的 `make test-all` 结果
- base 为 `main` 时记录当前 head 的远程 CI；其他 base 记录本地 gate
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
## UniSim extraction roadmap #1428

物理后端的新 public contract 位于独立的 `unisim-core` distribution（import
namespace 为 `unisim`）。UniLab 在迁移窗口内通过
`unilab.base.backend.unisim_bridge` 传递 task-owned `SceneCfg`；该 bridge
只负责边界翻译，不复制引擎实现。新增 consumer 应优先使用 bridge 或直接
调用 `unisim`，不要访问 backend 的 model/data 私有对象。迁移完成后
Child 12 会删除旧实现与兼容层。
