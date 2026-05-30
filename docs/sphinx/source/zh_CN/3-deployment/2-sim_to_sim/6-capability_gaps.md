# 能力缺口

语言: 简体中文

一张持续维护的表格，记录通过 `SimBackend` 及当前后端实现暴露出来的后端能力。
当某项能力被加入或从 backend contract 中移除时，请更新本页。

```{list-table}
:header-rows: 1
:widths: 30 15 15 40

* - 能力
  - MuJoCo
  - Motrix
  - 备注
* - `supports_physics_state_playback`
  - 是
  - 否
  - 由 `get_play_capabilities()` 上报。
* - `supports_native_interactive_renderer`
  - 否
  - 是
  - 由 `get_play_capabilities()` 上报。
* - `supports_native_video_capture`
  - 否
  - 是
  - 由 `get_play_capabilities()` 上报。
* - `create_hfield_scanner(...)`
  - 是
  - 是
  - 后端拥有的高度场扫描器，用于崎岖地形观测。
* - `apply_interval_randomization(...)`
  - 是
  - 是
  - 受支持的字段仍取决于各后端的 `get_dr_capabilities()`。
* - 位置执行器增益 DR
  - 是
  - 有条件
  - Motrix 通过运行时能力检测上报支持情况。
* - Geom 摩擦 DR
  - 是
  - 有条件
  - Motrix 通过运行时能力检测上报支持情况。
```

把 `src/unilab/base/backend/base.py` 作为 contract 的来源，把 `tests/base/`
下的后端测试作为支持声明的证据。

## 如何更新本页

当你添加或移除某项能力时：

1. 更新上面的表格。
2. 如果该变更引入了新的 contract，则在 {doc}`/adr/ADR-0000-index` 下添加或更新
   相关的 ADR。
3. 在 `tests/base/` 下添加或更新后端测试。
4. 在 changelog 中提及对用户可见的变更（参见 {doc}`/changelog`）。

## 另请参阅

- {doc}`1-backend_swap`
- {doc}`../../4-developer_guide/2-contracts/2-backend_contract`

## Navigation

- Index: [文档](0-index.md)
