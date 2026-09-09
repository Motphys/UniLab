---
orphan: true
---

# ADR-0008 Debug Overlay Primitive Contract And Embeddable Playback Session

- Status: Accepted
- Date: 2026-09-09
- Owners: Backend / Visualization maintainers
- Supersedes: None
- Superseded by: None

## Context

播放/录制管线的任务侧叠加层过去依赖 `(num_envs, 3)` marker 位置数组
（`extra_data_getter`），只能表达“每个 env 一个球”，无法表达坐标系、箭头、
ghost mesh 或文本等任务 debug 语义，交互 viewer（`play_interactive.py`）与离线
record 管线各自维护了一套互不兼容的绘制代码。

上游 unisim（unilabsim/unisim#53, 驱动 issue unilabsim/wuji_unilab#21）把
`SimBackend.run_playback` 的 `extra_data_getter` 替换为
`debug_overlay_getter`，引入 typed `DebugPrimitive`
（sphere/box/frame/arrow/ghost_geom/text，env 局部系位姿）、typed `CameraCfg`
（`from_kwargs` 归一化、未知键 fail-closed）以及
`BackendPlayCapabilities.supports_debug_overlay`。UniLab 作为下游需要：

1. 把 env 契约（`ABEnv`/`NpEnv`）和所有调用点迁移到新契约；
2. 收敛交互 viewer 的硬编码 `user_scn` 叠加层到同一原语契约；
3. 把 record 管线的“快照缓存 + 事后渲染”暴露为可嵌入组件，供自定义 eval
   循环（自写 trial 协议）复用。

这跨越 env/backend 公共契约，普通实现说明不足以作为 review 基线。

## Decision

1. **叠加层一律使用 typed `DebugPrimitive`。** env 契约透传
   `debug_overlay_getter: Callable[[], Sequence[Sequence[DebugPrimitive] | None] | None]`，
   外层长度等于 `num_envs`，位姿为 env 局部系；grid offset 由渲染器应用。
   旧的 `extra_data_getter` marker 数组契约在 UniLab 侧零残留。
2. **capability 门控 fail-closed。** `EnvPlayCapabilities` 增加
   `supports_debug_overlay` 并透传 backend capability；调用方在 backend 不
   支持时省略 getter，backend 在收到 getter 但不支持时抛
   `NotImplementedError`。
3. **交互 viewer 与离线渲染共用原语数据层。** `play_interactive.py` 只构造
   `DebugPrimitive` 列表；注入 `viewer.user_scn` 的统一入口是
   `unilab.visualization.debug_primitives.append_debug_primitives_to_scene`。
   该注入器是 UniLab 持有的过渡实现——unisim 的离线 worker 绘制实现
   （`_append_primitive`/`_append_debug_primitives`）是私有的，且面向多 env
   网格合成而非交互路径（单 env、已加载 model、调用方持有 mjvScene）；
   待 unisim 暴露适合交互路径的公开 helper 后改为委托（见 Consequences）。
4. **record 管线组件化为 `SnapshotPlaybackSession`。**
   `unilab.visualization.playback_session.SnapshotPlaybackSession` 把
   `snapshot()`（每 trial 步缓存 physics state 与可选叠加层）与
   `render_snapshots(output_video=..., overlay_getter=..., camera=..., fps=...,
   on_frame=...)`（统一渲染出 mp4）暴露为 session 级公开操作，渲染走共享的
   MuJoCo 离线 snapshot 管线。session 只持有 NumPy 数组与 typed 原语，缓存
   可 pickle；trial 之间用 `clear()` 界定生命周期。
5. **`ABEnv.render(mode="rgb_array")` 便捷封装。** 内部走
   `init_play_renderer(headless=True, capture=True)` +
   `capture_play_video_frame`，按 `supports_native_video_capture` 门控，不支持
   时抛带类名的 `NotImplementedError`。
6. **camera 配置面统一过 `CameraCfg.from_kwargs`。** train/play 脚本通过
   `unilab.visualization.playback.camera_cfg_from_training` 组装 typed camera
   配置；Hydra YAML 字段名不变，未知键在边界 fail-closed。
7. **`on_frame` 回调先在 session 层落地。** `run_playback`/`run_playback_mode`
   在 env 契约上声明 `on_frame: Callable[[int, np.ndarray], np.ndarray | None]`，
   但 unisim `SimBackend.run_playback` 尚未声明该参数；env 透传层在收到
   `on_frame` 时 fail-closed 抛 `NotImplementedError`，实际帧回调由
   `SnapshotPlaybackSession.render_snapshots(on_frame=...)` 提供，待上游补契约后
   再透传（见 Consequences）。
8. **任务自有 overlay 走统一发现入口。** `ManagerBasedRlEnv` 提供
   `get_playback_debug_overlays()`：遍历 command manager 的 terms，聚合实现了
   `playback_debug_overlay_getter()` 的 term，把多 term 的原语按 env 合并成单个
   `DebugOverlayGetter`；无 provider 时返回 `None`。play 入口
   （`train_rsl_rl.play_rsl_rl`、`play_interactive.py`）通过
   `getattr(env, "get_playback_debug_overlays", None)` 发现任务 overlay，发现不到
   时回退现有特例（`curr_ee_goal_world` EE goal sphere、交互 viewer 的
   motion/reward/velocity 硬编码），既有行为不回归。
9. **overlay 门控按 resolve 后的渲染模式区分。** 落点在
   `ABEnv.run_playback_mode`（`on_plan` 回调之后、dispatch 之前，plan 对象不被
   改写）：record 模式要求 `supports_debug_overlay`，backend 不支持时维持
   fail-closed；interactive 模式要求
   `supports_interactive_debug_overlay`（unisim#54 起 mjwarp 报 True，其余
   backend 为 False），不满足时 `warnings.warn` 并丢弃 getter（传 None），
   交互回放照常进行——任务 opt-in overlay 不应让无位 backend 的交互回放整体
   不可用。

## Stable Contracts

- 任务侧叠加层的唯一数据契约是 `unisim.backend.base.DebugPrimitive` 列表
  （per-env、env 局部系）；不允许再引入按位置数组或 backend 私有 geom 操作
  的叠加层入口。
- `EnvPlayCapabilities.supports_debug_overlay` 是 env 侧判断叠加层可用性的
  唯一入口；interactive 模式的叠加层可用性由
  `supports_interactive_debug_overlay` 表达，门控统一在 `run_playback_mode`。
- 任务自有 overlay 的唯一发现入口是 `ManagerBasedRlEnv.get_playback_debug_overlays()`
  （command terms 实现 `playback_debug_overlay_getter()`）；play 入口用
  `getattr` 发现并回退现有特例，不再新增绕过发现机制的任务特例。
- 自定义 eval 循环通过 `SnapshotPlaybackSession` 复用 record 管线，不在脚本里
  重新实现“快照缓存 + 事后渲染”。
- camera 参数在 train/play 入口统一经 `camera_cfg_from_training` /
  `CameraCfg.from_kwargs` 归一化。

## Alternatives Considered

- 在 UniLab 复制 unisim 离线 worker 的 `_append_primitive` 实现供交互路径使用。
  拒绝原因：跨仓库复制同一绘制实现必然漂移；交互路径只需要
  sphere/box/frame/arrow 的 mjvScene 注入，过渡实现保持最小并明确等待上游
  公开 helper。
- 交互 viewer 继续使用硬编码 `user_scn` 绘制、只迁移 record 管线。拒绝原因：
  两条路径的 overlay 语义会继续分叉，task 侧 debug 可视化无法在两种
  渲染路径间复用。
- 自定义 eval 循环直接调用 `env.run_playback()` 并自行包一层 trial 协议。
  拒绝原因：`run_playback` 是单体的“initialize/step/渲染”整体入口，无法
  表达“逐 trial 攒帧、trial 结束统一出片”的协议，会导致下游复制 record
  管线内部逻辑。

## Consequences

- `extra_data_getter` 在 UniLab 全仓（含 docstring/tests）零残留；新叠加层
  代码必须使用 `DebugPrimitive`。
- 交互注入器 `unilab/visualization/debug_primitives.py` 是过渡组件，当
  unisim 提供适合交互路径（单 env、已加载 model、调用方持有 mjvScene）的
  公开 helper（建议形态：`append_debug_primitives(scene, overlays, *,
  offsets=None, mesh_ids=None)`）后应改为委托并删除本地实现。
- `on_frame` 透传依赖 unisim 在 `SimBackend.run_playback` 上声明同名参数；
  上游落地前 env 层 fail-closed。
- UniLab 依赖 unisim PR #53 的契约；该 PR 合并并发版后需 bump
  `unisim-core` 最低版本。

## Evidence In Repo

- env 契约: `src/unilab/base/base.py`, `src/unilab/base/np_env.py`
- 任务 overlay 发现: `src/unilab/envs/manager_based_rl_env.py`（`get_playback_debug_overlays`）
- 交互注入器: `src/unilab/visualization/debug_primitives.py`
- 可嵌入 session: `src/unilab/visualization/playback_session.py`
- camera 归一化: `src/unilab/visualization/playback.py`
- 交互 viewer 迁移: `src/unilab/scripts/play_interactive.py`
- 训练入口迁移: `src/unilab/scripts/train_rsl_rl.py`, `src/unilab/scripts/train_appo.py`, `src/unilab/scripts/train_offpolicy.py`, `src/unilab/scripts/play_hora_appo.py`
- 上游契约: `unisim.backend.base`（`DebugPrimitive`, `CameraCfg`, `DebugOverlayGetter`, `validate_debug_overlays`, `BackendPlayCapabilities.supports_debug_overlay` / `supports_interactive_debug_overlay`）
- 测试: `tests/visualization/test_debug_primitives.py`, `tests/visualization/test_playback_session.py`, `tests/base/test_np_env_playback_contract.py`, `tests/envs/test_manager_based_rl_env.py`（overlay 聚合）

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`ADR-0002 Backend Capability Boundary For Play And Snapshot </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
