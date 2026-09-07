# 协作工作流

本页是仓库关于 issue 范围、roadmap 分支、PR、架构决策与发布 gate 的唯一规则来源。
稳定规则写在这里；执行状态、负责人和交付跟踪写入 GitHub issue 与 PR。

## 与 Agent 协作

Agent 应先确认请求的交付结果，检查 owner layer、调用方、测试与配置，再完成最小但完整的
改动。日常实现选择和已授权的可逆工作可以持续推进；只有答案会改变产品方向、公共 contract、
不可逆外部动作或长期维护成本时，才提出聚焦的问题。

研究、测试、日志分析或 review 能彼此独立时，可并行委派这些读密集工作，并向主任务返回
精简证据。同一文件集不得并发写入。实施 agent 仍负责整合证据、验证最终工作树并报告结果。

## 工作项

一个 issue 交付一个可观察、可审查的结果。仅在作为完整纵向切片时才有价值的代码、配置、
测试和文档应放在同一个 issue。一个子项拥有独立验收标准、owner 决策、回退路径或独立价值时，
将其拆为 child issue。

每个 implementation issue 记录：

1. 问题及仓库证据。
2. 交付后的行为与明确范围边界。
3. 受影响的 owner layer 与 contract。
4. 验收标准和贴近风险的验证。
5. 依赖、目标 PR base，以及重要的平台/后端影响。

Roadmap 只用于需要多个可独立审查的 child PR 才能完成的集成结果。Roadmap 开头提供面向
maintainer 的简短说明：目标、推荐路径、交付边界、预计 review 规模、长期责任与待决事项。
Research 或 benchmark 的交付物是可复现证据和决策结论，不自动构成生产 support 承诺。

## 范围决策

开发授权覆盖已确认的 issue 或 roadmap 边界。新增公共 contract、execution path、runner/env
lifecycle、同步协议、常规 CI、support 承诺、长期 benchmark 设施、历史重写或 production
subsystem 前，须由 maintainer 决策。证据改变预计规模、依赖、owner 边界，或证明上游/更小方案
可行时，也要复核范围。扩展工作前把该决定记录到 issue、roadmap 或 ADR。

## Roadmap 分支

Roadmap 获得开发授权后，记录 declared base，并从其最新 head 创建
`dev/issue-<roadmap-number>-<slug>`。base 可以是 `main` 或上层 roadmap 集成分支。每个
child branch 从最新集成分支创建，使用常规类型前缀，例如
`feat/issue-<number>-<slug>` 或 `fix/issue-<number>-<slug>`，并以集成分支为 PR base。

进入 review 前，将 child 同步到当前集成分支，并在同步后的 head 验证。child PR 合入后，再次
验证集成分支 head，然后创建回到 declared base 的最终 PR。declared base 前进时，在计划的
集成点同步，并重新运行相应 gate。

## PR 与 CI

开发前选择并记录目标 base。每个 PR 关联驱动它的 issue，说明交付行为和影响，并列出在最终
本地 head 实际运行的命令。下表定义所需本地与远程证据；PR 模板记录证据。

| PR base | 合入前的必需证据 |
| --- | --- |
| `main` | 最终 head 的贴近风险检查和 `make test-all`；当前 head 的所有适用远程 CI 均通过。 |
| 其他分支 | 最终 head 的贴近风险检查和 `make test-all`；本地结果和 review 构成 gate。后续进入 `main` 的 PR 运行远程 CI。 |

使用 Conventional Commit 标题。说明改动是否影响 MuJoCo、Motrix、其他后端、macOS、Linux 或
训练行为。文档改动运行相应的 docs 检查，并遵守该 PR 所需的仓库 gate。

## 架构决策

改动 runtime、backend、config、registry 或其他公共 contract 时，在 issue 或 PR 关联相应 ADR。
只有现有 ADR 无法覆盖新的结构性决策时，才在同一 PR 新增 ADR。使用
{doc}`ADR 模板 </adr/ADR-TEMPLATE>`，包括状态、owner、替代/取代关系、备选方案、仓库证据和
关联文档。

## 发布

从已批准的最终 commit 发布：更新 `pyproject.toml` 中唯一版本，运行 `make test-all` 和
`uv build`，并确认该 commit 存在成功的 `ci.yml`。推送与项目版本一致的 annotated
`v<version>` tag。release workflow 构建并验证发行包，再通过 PyPI trusted publishing 发布
带 tag 的构建。发布失败后若改动代码，必须使用新版本和新 tag；不得覆盖已发布版本。
