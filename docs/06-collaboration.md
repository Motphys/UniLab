# Collaboration Workflow

仓库文档写稳定标准；执行状态、owner 和阶段推进放在 GitHub 协作对象里。

如果你只是想安装或训练 UniLab，请先看 `README.md`、`docs/01-getting-started.md` 和 `docs/03-training.md`。

## Work Item Granularity

每个 issue 至少回答以下问题：

1. 我们在解决什么问题？
2. 预期交付物是什么？
3. 完成标准是什么？
4. 谁负责执行？
5. 有哪些上游阻塞？

推荐 issue 类型：

- `bug`
- `work item`：feature / infra / benchmark / test / sim / docs work

## Milestone Structure

每个 milestone 应该：

- 在 GitHub 上创建一个 milestone 对象
- 创建一个 tracking issue 汇总子 issue
- 把执行细节写在子 issue，不写在 milestone 描述里
- 以产物定义完成，而不只是“代码已 merge”

典型完成产物：

- green CI
- benchmark 结果或 W&B run link
- demo video / ONNX export / checkpoint path
- 若用户可见行为变化，则附带 docs update

## PR Evidence Standard

每个 PR 应该：

- link 对应 driving issue
- 描述 user-facing change 和 training-impact
- 列出实际执行过的 validation commands
- 说明行为是否在 `mujoco`、`motrix`、macOS、Linux 间发生变化

## Ownership Model

使用 GitHub assignees 表达执行 owner，使用 `CODEOWNERS` 表达 review owner。
如果负责人暂时还没有稳定的 GitHub handle，就先保持 issue unassigned，并在 issue body 中临时记录预期 owner。

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](../CONTRIBUTING.md)
