# 协作流程

语言: [English](../en/06-collaboration.md) | 简体中文 | [日本語](../ja/06-collaboration.md) | [한국어](../ko/06-collaboration.md)

仓库文档只记录稳定标准。执行状态、owner 和阶段推进应放在 GitHub 协作对象中。

如果你只是想安装或训练 UniLab，请先看 `docs/zh_CN/README.md`、`docs/zh_CN/01-getting-started.md` 和 `docs/zh_CN/03-training.md`。

## Work Item Granularity

每个 issue 至少应该回答这些问题:

1. 我们在解决什么问题？
2. 预期交付物是什么？
3. 完成标准是什么？
4. 谁负责执行？
5. 有哪些上游阻塞？

推荐的 issue 类型:

- `bug`
- `work item`: feature / infra / benchmark / test / sim / docs work

## Milestone Structure

每个 milestone 应该:

- 在 GitHub 中作为 milestone 对象存在
- 有一个 tracking issue 来汇总子 issue
- 把执行细节写在子 issue 里，而不是 milestone 描述里
- 用交付产物来定义完成，而不只是“代码已合并”

典型完成产物:

- green CI
- benchmark 结果或 W&B run 链接
- demo video / ONNX export / checkpoint path
- 若用户可见行为变化，则附带 docs 更新

## PR Evidence Standard

每个 PR 都应该:

- 链接 driving issue
- 描述用户可见变化和训练影响
- 列出实际执行过的验证命令
- 说明行为是否在 `mujoco`、`motrix`、macOS 或 Linux 间发生变化

## Ownership Model

执行 owner 用 GitHub assignees 表达，review owner 用 `CODEOWNERS` 表达。如果暂时没有稳定的 GitHub handle，就让 issue 保持 unassigned，并在 issue body 里临时注明预期 owner。

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](CONTRIBUTING.md)
