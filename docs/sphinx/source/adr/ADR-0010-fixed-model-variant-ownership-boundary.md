---
orphan: true
---

# ADR-0010 Fixed Model Variant Ownership Boundary

语言: 简体中文

- Status: Accepted
- Date: 2026-09-13
- Owners: Env / Config / Backend maintainers
- Supersedes: None
- Superseded by: None

## Context

[Discussion #1541](https://github.com/Motphys/UniLab/discussions/1541)
指出，legacy domain-randomization provider 与 SimToolReal 大量工具模型瓶颈有共同根因：
backend 缺少 per-env model identity / model-field indirection。因此 task 曾被迫在
UniLab 侧编译多个 engine model，或在 reset 协议中扩展模型变更语义。

当前分层已经明确：

- `mjbatch-uni` 拥有 CPU MuJoCo batch executor、same-layout mesh pooling 与
  topology-affine routing；
- UniSim 拥有 `SimBackend`、DR capability、backend adapters 与 engine-native
  realization；
- UniLab 拥有 Hydra task owner config、EventManager/reset transaction 与 task identity。

本决策固定 fixed model/tool variant 的跨仓边界，避免后续 contract child 把
executor 细节或 live engine objects 上移到 task 层。

## Decision

### Ownership

UniLab 只拥有任务选择语义：

- 声明 named fixed model/tool variant source catalog 与 optional explicit names；
- 在冷路径直接生成 UniSim construction-time plan；
- 保持 assignment 在 backend construction/materialization 后不可变；
- 不打开、解析或编译 variant source，不持有 `MjSpec`、`MjModel`、mjbatch object、
  Warp array 或任何 backend-private handle。

UniSim 拥有唯一 public negotiation/realization contract：

- 在既有 `DomainRandomizationCapabilities` 上声明 fixed-variant capability；
- 定义 pickle-safe 的 construction-time variant plan；
- 将 UniLab source descriptor 翻译为 backend-family preparation input；
- MuJoCo/MJWarp adapter 分别选择 executor realization；
- 声明 per-env playback 语义。

mjbatch-uni 与 MJWarp/mjlab 路径只作为 UniSim adapter 的 engine implementation。
same-layout compiler coherence、mesh dedup、per-world arrays、CUDA graph capture 前
初始化、derived constants 与 per-env playback 均不得成为 UniLab API。

### Task Configuration And Assignment

`EnvCfg.fixed_model_variants` 是 task owner 的声明性 catalog。每个 entry 只有
name 与 source path descriptor；空 `explicit_variant_names` 选择 deterministic
round-robin，非空列表表示 task 已经展开的 exact assignment。Manager factory 直接
生成携带 read-only `int32`、形状 `(num_envs,)` final index array 的 UniSim plan；
不公开第二个 materialization 对象。backend-local copies 可以存在，但不能改写
task final identity。

Assignment 是 construction-time task identity，不在 reset 时重采样。reset event terms
只能在既有 model identity 内提交 curated model-field payload；mesh/tool identity 的
变更必须重新构造 backend。

当前 schema 只承诺 same-public-layout variants：`nq`、`nv`、actuator/action shape、
sensor layout 与 observation contract 必须一致。无法投影到统一 public layout 的
heterogeneous topology 在本决策中 fail closed，等待独立 contract。

### Capability Negotiation

fixed-variant 支持必须由 UniSim 的 DR capability object 显式声明为
`supports_fixed_variants`。缺失与 `False` 等价并且 fail closed；UniLab 不得根据
backend 名称、可选 package import、executor introspection 或异常降级推断支持。

在 UniSim U1/U2/U3 落地前，配置了 fixed variants 的 Manager-Based env 在创建 env
前失败，并清理已创建的 unsupported backend。legacy
`DomainRandomizationManager` / provider protocol 不获得该能力。

## Stable Contracts

- Task catalog/final assignment: `src/unilab/base/variants.py`
- Owner config entry: `EnvCfg.fixed_model_variants`
- Hydra typed materialization: `src/unilab/base/config_materialization.py`
- Fail-closed Manager lifecycle guard: `src/unilab/envs/manager_based_rl_env.py`
- Contract tests: `tests/base/test_fixed_model_variants.py`
- UniSim extraction boundary: [ADR-0007](ADR-0007-unisim-extraction-boundary.md)

## Alternatives Considered

- 让 task config 直接持有或返回 `MjSpec`。拒绝原因：task YAML 变成 MuJoCo-specific，
  UniSim 难以保持 MJWarp 兼容，且 public plan 不再 pickle-safe。
- 在 UniLab 为每个 env 编译完整 model。拒绝原因：这正是 SimToolReal 内存和冷启动
  瓶颈，并把 engine realization 上移到错误 owner。
- 在 reset provider 中切换 model identity。拒绝原因：破坏派生常量、CUDA graph 和
  playback 的生命周期假设，也会延长 legacy DR 协议共存。
- 暴露通用 dict/field-name mutation API。拒绝原因：engine schema 与 executor details
  会泄漏成 public contract，难以跨 MuJoCo/MJWarp 保持版本兼容。

## Consequences

- Task owner 可以描述 600 个固定工具及最终 assignment，但实现无需把 600 个 live model
  带入 UniLab。
- UniSim contract child 必须先提供 capability 与 construction-time plan，再让 task
  rollout child 消费；不能要求 UniLab import `mjbatch`。
- MuJoCo CPU 与 MJWarp 可以使用不同 realization，但必须接受同一个 neutral source
  descriptor/final assignment，并保留各自 capability 差异。
- Fixed identity 与 reset-time model-field DR 分离：前者 immutable，后者由
  Manager-Based reset transaction 一次性提交。
- Manager model-field 默认值只能来自 UniSim 的
  `SimBackend.get_reset_term_default(term)`；canonical 与 per-world 表由 backend
  权威返回，UniLab 不再为了 `body_inertia` 重新编译 MuJoCo scene。

## Evidence In Repo

- `src/unilab/base/variants.py`
- `src/unilab/base/base.py`
- `src/unilab/base/config_materialization.py`
- `src/unilab/envs/manager_based_rl_env.py`
- `src/unilab/base/reset_state.py`
- `src/unilab/base/entity.py`
- `tests/base/test_fixed_model_variants.py`

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`Domain Randomization </zh_CN/2-user_guide/5-domain_randomization/0-index>`
- [Roadmap #1563](https://github.com/Motphys/UniLab/issues/1563)
