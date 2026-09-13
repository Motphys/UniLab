# 域随机化 Contract

Manager-Based event term 是 UniLab 唯一的 DR lifecycle。任务 provider 协议已移除，
`NpEnv` 不再携带 DR manager。

## 生命周期

- **Construction identity：** `env.fixed_model_variants` 物化最终 read-only
  assignment，并把 UniSim `FixedVariantPlan` 附到 `SceneCfg`。backend 在 first
  forward 与 CUDA graph capture 前完成 realization。
- **Reset：** event term 通过 Entity binding 写入 `ResetStateTransaction`；
  transaction 只调用一次 `SimBackend.set_state(..., randomization=...)`。
- **Interval：** event term 通过公开 `SimBackend` contract 使用 backend-owned
  interval plan。

## 能力边界

Backend 差异是显式 capability，不是 task-side 分支：

- `DomainRandomizationCapabilities.supported_reset_terms`
- `supported_interval_terms`
- fixed-variant layout
- per-environment playback 支持

未声明支持的请求 term 会携带 backend 与 term 名称 fail closed。Manager code 不
import MuJoCo 或 mjbatch，也不访问 backend model/pool。

## Reset Payload 与默认值

`ResetRandomizationPayload` 是 curated NumPy plan，首维为 selected row count。
支持项包括 body mass/COM/inertia family、gravity、geometry friction/size/solver
参数、joint damping/armature/friction 与 actuator gains。geometry bounds 等
派生字段归 backend 所有，caller 不能独立提交。

冷路径 binding 时，`ResetStateTransaction` 向 UniSim 请求
`SimBackend.get_reset_term_default(term)`。返回表格是权威默认值，只有两种布局：

- canonical model table，例如 `body_mass` 的 `(nbody,)`；
- per-environment fixed-variant table，例如 `(num_envs, nbody)`。

对 selected reset subset，event term 使用对应 env rows 作为 baseline。只写部分
model columns 时，transaction 会用同一 env row 填充未写列，再构造一个 dense
payload。缺失能力、未支持 term、非 floating 表、非法 tail、首维不是 `num_envs`
的 per-env 表均 fail closed。该边界移除了 UniLab 侧为 inertia 默认值重新 compile
MuJoCo XML 的路径。

## Interval Terms

Interval plan 基于 term descriptor：`IntervalRandomizationPlan.ops` 携带来自
`unisim.dr.interval` 的 `IntervalTermOp`。内置 payload contract 由
`IntervalTermOp.validate` 强制；未知 backend-owned custom term 传给该
backend handler table。Ops 与 plans 保持 stdlib/NumPy 数据，可跨 spawn collector
pickle。

## 仓库证据

- Manager lifecycle：`src/unilab/managers/event_manager.py`
- Reset transaction：`src/unilab/base/reset_state.py`
- Entity bindings：`src/unilab/base/entity.py`
- Task-owned fixed variants：`src/unilab/base/variants.py`
- Backend contract/capability types：`unisim.backend.base`、`unisim.dr.types`
- ADR：{doc}`ADR-0010 Fixed Model Variant Ownership Boundary </adr/ADR-0010-fixed-model-variant-ownership-boundary>`
