# 域随机化 Contract

Manager-Based event term 是 UniLab 唯一的 DR lifecycle。任务 provider 协议已移除，
环境不再携带 DR manager。

## 生命周期

- **Construction identity：** `env.fixed_model_variants` 物化最终 read-only
  assignment，并把 UniSim `FixedVariantPlan` 附到 `SceneCfg`。backend 在 first
  forward 与 CUDA graph capture 前完成 realization。
- **Startup：** Manager 构造完成后，backend 先 materialize，event term 再采样
  一个 full-width per-environment 表，并在 read-plan 绑定、第一次 reset 或 step
  之前复用与 reset DR 相同的 backend-owned packed reset 事务
  （`SimBackend.bind_packed_reset()`）提交。提交成功后，packed composer 将
  已提交行的 staged 字段提升为其持久 reset baseline，因此后续 selected-row
  reset 会保留未写入的 startup 值。Startup 是显式 opt-in，用连续分布覆盖换取
  steady-state reset 路径中的 full-model derived-constant refresh。
- **Reset：** event term 通过 Entity binding 写入 `ResetStateTransaction`；
  transaction 将全部 staged 写入组合进 backend-owned packed 事务，并恰好提交
  一次。
- **Range 写法：**生命周期选择只改变采样边界。owner YAML 中标量范围沿用
  reset owner 的数组形式，例如 `mass_distribution_params: [lo, hi]`；mass/COM
  家族在 startup 与 reset 声明中都保持轴映射，例如
  `com_range: {x: [lo, hi], y: [lo, hi], z: [lo, hi]}`。
- **Interval：** event term 通过公开 `SimBackend` contract 使用 backend-owned
  interval plan。

## 能力边界

Backend 差异是显式 capability，不是 task-side 分支：

- `DomainRandomizationCapabilities.supported_reset_terms`——backend 通过公开 DR
  路径支持的全部 reset term
- `supported_interval_terms`——backend 通过公开 DR 路径支持的全部 interval term
- fixed-variant layout
- per-environment playback 支持
- `TensorLifecycleCapabilities.packed_reset` 与 `SimBackend.bind_packed_reset()`：
  Manager-Based reset 与 startup DR 生命周期的必需 capability；缺失声明会在 env
  构造时 fail closed。

这两个集合是单一且与 plane 无关的 term 集合。Manager capability 协商不检查
payload 所在设备，也不传递 plane。CPU/GPU 执行及所需的内部转换由 backend
adapter 负责，不再拆成不同的公开 capability 集合。未声明的请求 term 会携带
backend 与 term 名称 fail closed；backend 仍会拒绝违反其设备或生命周期契约的
payload。Manager code 不 import MuJoCo 或 mjbatch，也不访问 backend model/pool。

## Reset Payload 与默认值

`TensorResetRandomizationPayload` 是唯一公开的 reset payload：curated torch
plan，首维为 selected row count。其 16 个字段覆盖 body mass/COM/inertia family、
gravity、geometry friction/size/solver 参数、joint damping/armature/friction、
actuator gains，以及 `base_mass_delta`/`base_com_offset` delta 项。字段保持
`Any` 注解与惰性 torch 校验，因此 `unisim-core` 维持仅依赖 NumPy 的打包契约。
NumPy 仅保留在 adapter 私有的 host 转换（`unisim.dr._host`）、冷元数据与
serialization/playback leaf 之后；原先的 NumPy `ResetRandomizationPayload` 已
不存在。geometry bounds 等派生字段归 backend 所有，caller 不能独立提交。

packed composer 在 commit 时把 staged delta 项与权威 baseline 在 torch 中合成为
最终绝对值表（`final = baseline + delta`），因此 CUDA 环境的 reset DR 不再经
host 往返。

冷路径 binding 时，`ResetStateTransaction` 向 UniSim 请求
`SimBackend.get_reset_term_default(term)`。返回表格是权威默认值，只有两种布局：

- canonical model table，例如 `body_mass` 的 `(nbody,)`；
- per-environment fixed-variant table，例如 `(num_envs, nbody)`。

对 selected reset subset，packed composer 持有的 baseline 是权威来源。只写部分
model columns 时，只有这些列会被 stage；未写列保留 composer 的 per-env
baseline，并由 backend-owned composer 在唯一一次提交时合并。缺失能力、未支持
term、非 floating 表、非法 tail、首维不是 `num_envs` 的 per-env 表均 fail
closed。该边界移除了 UniLab 侧为 inertia 默认值重新 compile MuJoCo XML 的路径。

## Interval Terms

Interval plan 基于 term descriptor：`IntervalRandomizationPlan.ops` 携带来自
`unisim.dr.interval` 的 `IntervalTermOp`。payload 为 torch-first：
`IntervalTermOp.payload` 是 torch tensor 或 `TensorIntervalPayload`；push 在所有
后端统一为生产方采样——后端内部的全局 RNG push 采样已不存在。
内置 payload contract 由 `IntervalTermOp.validate` 强制；未知 backend-owned
custom term 传给该 backend handler table。跨 spawn collector 的 plan 携带采样
范围而不是 tensor；device payload 是进程本地的。

该结构性决策记录在 UniSim 的
[torch-first DR payload ADR](https://github.com/unilabsim/unisim/blob/main/docs/zh/adr-dr-torch-first-payloads.md)。

## 仓库证据

- Manager lifecycle：`src/unilab/managers/event_manager.py`
- Reset transaction：`src/unilab/base/reset_state.py`
- Entity bindings：`src/unilab/base/entity.py`
- Task-owned fixed variants：`src/unilab/base/variants.py`
- Backend contract/capability types：`unisim.backend.base`、`unisim.dr.types`
- ADR：{doc}`ADR-0010 Fixed Model Variant Ownership Boundary </adr/ADR-0010-fixed-model-variant-ownership-boundary>`
