# Domain Randomization 契约

Domain randomization 是一个 env-owner 的 provider 契约，加上 backend 能力的
应用。用户配置示例见
{doc}`../../2-user_guide/5-domain_randomization/0-index`。

## 生命周期分类

- Init 生命周期：改变模型身份或几何。这些改动在 env/backend 初始化、
  materialization 或 cache 构造期间执行。
- Reset 生命周期：在同一模型身份内改变状态或参数。Provider 通过 `ResetPlan`
  分发一份 reset 随机化 payload。
- Interval 生命周期：在两步之间施加扰动，例如 push 或 body force plan。

热路径不得解析 XML/资源，也不得用 `getattr` 或 `hasattr` 探测 backend 私有方法。

## Provider 最低要求

使用 DR 的任务应定义：

1. 一个由任务拥有的 domain-randomization config dataclass。
2. 一个 `DomainRandomizationProvider`。
3. 返回 `ResetPlan` 状态与随机化 payload 的 reset 行为。
4. 必要时通过 `IntervalRandomizationPlan` 实现的 interval 行为。
5. 在 env 构造中调用 `self._init_domain_randomization(...)`。

共享类型位于 `unisim.dr.types`（interval term 描述符位于
`unisim.dr.interval`），两者都由 `src/unilab/dr/__init__.py` 再导出。
manager 行为位于 `src/unilab/dr/manager.py`。

## Backend 能力边界

Backend 支持是显式的。只有当以下三个部分同时存在时，一个 reset 或 interval
条目才算作统一的 DR 条目：

1. `ResetRandomizationPayload` 中有明确的字段，或
   `IntervalRandomizationPlan.ops` 携带该 term 的 `IntervalTermOp`。
2. backend 声明并实现了该能力。
3. 任务 config/provider 对该字段或 op 进行采样并分发。

MuJoCo 与 Motrix 的差异保留在 backend 能力声明、backend 实现与 owner YAML 中。

## Interval Term 描述符

Interval plan 基于 term 描述符：`IntervalRandomizationPlan.ops` 携带一个
`IntervalTermOp` 元组（term 名称、NumPy payload、可选的 `body_ids`），定义在
`unisim.dr.interval`，并经 `unilab.dr` 再导出。

- 内置 term 名称即 `INTERVAL_TERM_*` 常量；其 payload 契约由
  `INTERVAL_TERM_SPECS` 固定（`push`：payload 形状 `(3,)`，不接受
  `body_ids`；四个 body term：payload 形状 `(num_envs, len(body_ids), 3)`，
  且必须携带 `body_ids`）。`IntervalTermOp.validate()` 对内置 term 强制
  这些契约；未知的自定义 term 原样通过校验。
- 能力所有权留在 backend：
  `DomainRandomizationCapabilities.supported_interval_terms` 是权威声明，
  通过 `supports_interval_term` / `get_unsupported_interval_terms` 查询。
- `DomainRandomizationManager.apply_interval_randomization_if_due` 是通用的：
  不包含任何 term 名称或按 term 分支，因此 backend 拥有的自定义 term 无需
  修改 manager。不在能力集合中的 term 会 fail-closed，抛出带有 backend 类型
  与 term 名称的 `NotImplementedError`；在 backend 一侧，
  `SimBackend.apply_interval_randomization` 把每个 op 路由到 handler 表，
  缺少 handler 时 fail-closed，抛出带有 backend 类名与 term 名称的
  `NotImplementedError`。
- Op 与 plan 必须保持 pickle 安全（protocol 4），以跨 spawn 方式的 collector
  子进程传递：只允许 stdlib + NumPy 的 frozen dataclass。
- 旧版 plan 字段（`push_perturbation_limit`、`body_ids`、
  `body_linear_velocity_delta`、`body_angular_velocity_delta`、`body_force`、
  `body_torque`）与旧版 `supports_interval_*` 能力布尔位已废弃：
  `IntervalRandomizationPlan.iter_ops()` 仍会把已设置的旧字段 1:1 适配为
  op，布尔位仍作为能力回退。新 provider 应填充 `ops`；旧字段将在下一个
  unisim-core major release 中移除。

## MuJoCo BatchEnvPool 快照

当前 MuJoCo 的 reset 随机化使用 `BatchEnvPool.reset(..., randomization=...)`，
并带有固定的字段白名单。带索引的读写可通过 `get_field_indexed(...)` 与
`set_field_indexed(...)` 实现。该接口位于 `mujoco-uni-runtime` 包
（`mujoco_uni.batch_env`），不在本仓库中；映射到它的 reset-term 常量定义在
`unisim.dr.types`。

支持的 reset 字段及其每 env 整块形状如下。首维始终是 `len(env_ids)`；尾部
整块大小是该字段在单个 `mjModel` 里的完整 flat 宽度。

| 字段 | 每 env 整块形状 |
| --- | --- |
| `body_mass` | `nbody` |
| `body_ipos` | `3 * nbody` |
| `body_iquat` | `4 * nbody` |
| `body_inertia` | `3 * nbody` |
| `dof_armature` | `nv` |
| `gravity` | `3` |
| `geom_friction` | `3 * ngeom` |
| `kp` | `nu` |
| `kd` | `nu` |

refresh 行为由 backend 固定：`body_mass`、`body_ipos`、`body_iquat`、
`body_inertia` 与 `dof_armature` 在写入后会触发 `mj_setConst` refresh，而
`gravity`、`geom_friction`、`kp` 与 `kd` 不触发。

两点注意：

- `geom_size` 不在 `SUPPORTED_FIELDS` 里。几何尺寸通过 init-lifecycle 的模型
  materialization 表达（见 `unisim.dr.types` 中的 `GeomSizeOverride` /
  `ModelVariantSpec`），不走 reset 随机化。
- `gravity` 的 reset 随机化需要包含它的 `mujoco-uni-runtime` 构建。本仓库依赖
  官方 `mujoco` 包（`>=3.5`，默认版本由 `uv.lock` 钉住）加 `mujoco-uni-runtime`，其 `SUPPORTED_FIELDS`
  包含 `gravity`；更旧的 batch-env 包（例如 `mujoco-uni==3.6.0.post6`）则没有。

## 电机控制扩展

对于不将策略输出直接映射到 backend 位置 actuator 的电机-actuator 任务，应将转换
保留在 env owner 层。通过 `SimBackend.set_pre_step_control(...)` 注册一个
pre-step 回调；backend 会在物理 substep 之前调用它，并在 stepping 之后刷新
sensor。

Go2W 是当前全电机 actuator 的示例：它的 env owner 将腿部位置目标与轮子力矩组合
在一起，而 kp/kd 随机化则保留在 env owner 的 cache 中，从而避免将 MuJoCo 位置
actuator 的机制泄漏到共享 payload 里。

## 仓库中的证据

- DR 类型：`unisim.dr.types` 与 `unisim.dr.interval`，由
  `src/unilab/dr/__init__.py` 再导出
- DR manager：`src/unilab/dr/manager.py`
- Backend 接口：`unisim.backend.base`
- 示例 provider：`src/unilab/tasks/locomotion/common/dr_provider.py`、
  `src/unilab/tasks/locomotion/go2_arm/manip_loco.py`、
  `src/unilab/tasks/manipulation/sharpa_inhand/rotation.py`
