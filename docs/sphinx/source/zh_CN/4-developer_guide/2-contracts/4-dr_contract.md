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

共享类型位于 `src/unilab/dr/types.py`，manager 行为位于
`src/unilab/dr/manager.py`。

## Backend 能力边界

Backend 支持是显式的。只有当以下三个部分同时存在时，一个 reset 或 interval
条目才算作统一的 DR 条目：

1. `ResetRandomizationPayload` 或 `IntervalRandomizationPlan` 中有明确的字段。
2. backend 声明并实现了该能力。
3. 任务 config/provider 对该字段进行采样并分发。

MuJoCo 与 Motrix 的差异保留在 backend 能力声明、backend 实现与 owner YAML 中。

## MuJoCo BatchEnvPool 快照

当前 MuJoCo 的 reset 随机化使用 `BatchEnvPool.reset(..., randomization=...)`，
并带有固定的字段白名单。带索引的读写可通过 `get_field_indexed(...)` 与
`set_field_indexed(...)` 实现。该接口位于 `mujoco-uni` 包
（`mujoco.batch_env`），不在本仓库中；映射到它的 reset-term 常量定义在
`src/unilab/dr/types.py`。

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
  materialization 表达（见 `src/unilab/dr/types.py` 中的 `GeomSizeOverride` /
  `ModelVariantSpec`），不走 reset 随机化。
- `gravity` 的 reset 随机化需要包含它的 `mujoco-uni` 构建。本仓库锁定
  `mujoco-uni==3.8.0`，其 `SUPPORTED_FIELDS` 包含 `gravity`；更旧的版本（例如
  `3.6.0.post6`）则没有。

## 电机控制扩展

对于不将策略输出直接映射到 backend 位置 actuator 的电机-actuator 任务，应将转换
保留在 env owner 层。通过 `SimBackend.set_pre_step_control(...)` 注册一个
pre-step 回调；backend 会在物理 substep 之前调用它，并在 stepping 之后刷新
sensor。

Go2W 是当前全电机 actuator 的示例：它的 env owner 将腿部位置目标与轮子力矩组合
在一起，而 kp/kd 随机化则保留在 env owner 的 cache 中，从而避免将 MuJoCo 位置
actuator 的机制泄漏到共享 payload 里。

## 仓库中的证据

- DR 类型：`src/unilab/dr/types.py`
- DR manager：`src/unilab/dr/manager.py`
- Backend 接口：`src/unilab/base/backend/base.py`
- 示例 provider：`src/unilab/envs/locomotion/g1/joystick.py`、
  `src/unilab/envs/motion_tracking/g1/tracking.py`、
  `src/unilab/envs/manipulation/sharpa_inhand/rotation.py`
