# 回放与快照差异

语言: 简体中文

除了物理步进本身之外，两个后端在**如何让你回放**一次运行这件事上也有所不同。
本页记录其中的实际影响。

## 回放

| 后端 | 机制 | 最适合 |
|---|---|---|
| MuJoCo | 由 `supports_physics_state_playback` 上报的物理状态回放路径 | 录制模式回放与离线视频导出 |
| Motrix | 由 `get_play_capabilities()` 上报的原生交互式渲染器与原生视频采集 | 交互式回放与录制模式采集 |

两个后端都通过 `SimBackend.resolve_play_render_plan(...)` 解析
`training.play_render_mode`；不受支持的模式应当在后端边界处失败，而不是在训练
脚本里分支。

## 快照

MuJoCo 目前上报 `supports_physics_state_playback=True`。而 Motrix 上报的是
原生交互式渲染与原生视频采集。请把它们当作不同的后端能力，而不是特性对等。

能力边界 contract
（{doc}`../../4-developer_guide/2-contracts/2-backend_contract`）要求 *env 代码*
和算法代码**都不得**直接调用仅快照可用的路径 —— 必须通过一个能力感知
（capability-aware）的抽象来路由。ADR-0002 将此固化为规范。

## 应该在任务 owner 里放什么

如果某个任务开始需要某种回放或快照能力，先把这一需求加入 backend contract，
然后在 env/backend 边界处验证它。不要把该需求藏在训练脚本的某个分支里。

## 另请参阅

- {doc}`6-capability_gaps`
- {doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`

## Navigation

- Index: [文档](0-index.md)
