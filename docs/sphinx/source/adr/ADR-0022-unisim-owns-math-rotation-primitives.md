---
orphan: true
---

# ADR-0022 UniSim Owns Math Rotation Primitives

语言: 简体中文

- Status: Proposed
- Date: 2026-11-05
- Owners: UniSim math primitives / UniLab base-runtime maintainers
- Supersedes: None
- Superseded by: None
- Roadmap: [Issue #2583](https://github.com/Motphys/UniLab/issues/2583)

## Context

四元数/旋转数学原语在迁移前同时存在于两个包的多个模块中。迁移前
`third-party/unisim/src/unisim/utils/rotation.py` 与 `src/unilab/base/rotation.py` 的
NumPy 实现逐行重复（`np_quat_mul`、`np_quat_mul_batched`、`np_quat_apply` 等函数体
一致），且两侧已出现漂移：unisim 侧有 `np_quat_conjugate_batched`、`np_yaw_from_quat`、
`np_wrap_to_pi`、`np_quat_inv`，unilab 侧则用本地 `_conjugate` 与
`np_quat_conjugate` 代替 `np_quat_inv`。Torch 侧同样存在多份并行实现：迁移前
`src/unilab/base/rotation.py`、`src/unilab/envs/mdp/observations.py`、
`src/unilab/tasks/motion_tracking/common/tensor_rotation.py` 各自定义了
`quat_apply_inverse` 等四元数算子。

重复实现不仅是维护成本问题，还带来训练热路径的 host 同步风险：
`src/unilab/base/entity.py` 的 `_host_root_link_lin_vel_b` /
`_host_root_link_ang_vel_b` 在 host(CPU) 侧用 NumPy 做世界系到体系的旋转，
误用会引入每步 D2H 拷贝与 CPU 同步。issue #2583 的审计（Child 4）确认训练热路径
（`src/unilab/envs/mdp/`、`src/unilab/tasks/`、`src/unilab/managers/`）对这两个
adapter 零调用，现有调用方仅为
`tests/base/test_entity_facade.py` 与
`tests/envs/mdp/test_uniform_velocity_command.py` 的显式测试路径。

## Decision

- unisim 是 backend-neutral 数学原语（math/rotation/transform）的唯一 owner。实现
  位于 `unisim/src/unisim/math/rotation_numpy.py`（NumPy）与
  `unisim/src/unisim/math/rotation_tensor.py`（Torch）。约定：四元数 wxyz 顺序、
  XYZ Euler 角顺序、`(..., 4)` 广播语义；Torch 热路径算子使用 `@torch.jit.script`。
- unilab 仅通过薄 re-export 包 `src/unilab/math` 消费这些原语，不保留任何兼容
  shim、别名或 deprecation fallback。迁移后 `src/unilab/base/rotation.py` 与任务内
  的本地旋转实现被删除或收缩为 re-export。
- 基础 wheel 保持 NumPy-only：`import unisim` 不加载 torch，Torch 实现按可选依赖
  惰性导入。
- 训练热路径禁止 host 侧旋转：`_host_root_link_lin_vel_b` /
  `_host_root_link_ang_vel_b` 仅限 backend/test 显式路径；`.cpu()`/`.numpy()`
  转移后立即调用四元数 helper 的 host 旋转同步模式在
  `src/unilab/envs/mdp/`、`src/unilab/tasks/`、`src/unilab/managers/` 中禁止，
  reset/startup 冷路径须逐文件登记豁免。

## Stable Contracts

- 四元数布局为 wxyz，Euler 为 XYZ 顺序，批量语义为 `(..., 4)` 广播；两个 backend
  实现共享同一语义，禁止出现第三份实现。
- `import unisim` 保持 NumPy-only，不引入 torch import；torch 算子仅在
  `unisim.math.rotation_tensor` 被显式导入时加载。
- unilab 侧只存在 `src/unilab/math` 薄 re-export；不在 unilab 内新增旋转/变换
  原语实现。
- 训练热路径对 host 根速度 adapter 零调用，由
  `tests/architecture/test_host_rotation_sync_boundary.py` 守护；冷路径豁免须在
  测试中登记并携带源码标记注释。

## Alternatives Considered

- **保留 unilab 本地实现、双份维护：** 拒绝。迁移前两侧已观察到实现漂移
  （`np_quat_inv` vs `np_quat_conjugate`、unisim 独有的 yaw/wrap helper），双份
  维护无法保证语义一致。
- **在 unilab 建唯一实现、unisim 反向依赖：** 拒绝。依赖方向颠倒；按
  ADR-0007 的抽取边界，backend-neutral 原语属于底层 unisim 包，unilab 是消费方。
- **通过兼容 shim/别名平滑迁移：** 拒绝。仓库 API 命名规则禁止别名与
  deprecation fallback；薄 re-export 已提供单一导入位置。

## Consequences

- 新增或修改旋转/变换原语只能在 unisim 侧进行，并同步更新 NumPy 与 Torch 两个
  实现及语义测试；unilab 侧 review 拒绝任何新本地实现。
- 迁移后 `src/unilab/base/rotation.py`、`src/unilab/envs/mdp/observations.py` 与
  `src/unilab/tasks/motion_tracking/common/tensor_rotation.py` 中的本地算子被删除
  或收缩为对 `src/unilab/math` 的引用；本 ADR Evidence 中的迁移前路径仅作历史
  证据。
- 训练热路径出现 `.cpu()`/`.numpy()` 后紧跟四元数调用的代码必须在 review 中
  拦截；确属冷路径的，在架构测试中逐文件登记豁免并加源码标记。
- 下游消费方（tasks/managers/mdp）统一从 `src/unilab/math` 导入，import 位置
  单一，便于后续审计。

## Evidence In Repo

- 迁移前重复实现：`third-party/unisim/src/unisim/utils/rotation.py` 与
  `src/unilab/base/rotation.py`（NumPy 函数体逐行重复，且已漂移）；Torch 侧
  `src/unilab/base/rotation.py::quat_apply_inverse`、
  `src/unilab/envs/mdp/observations.py::quat_apply_inverse`、
  `src/unilab/tasks/motion_tracking/common/tensor_rotation.py` 三份并行实现。
- Host 同步风险点：`src/unilab/base/entity.py`
  `_host_root_link_lin_vel_b` / `_host_root_link_ang_vel_b`（host 侧 NumPy 旋转）。
- 热路径零调用审计：全仓库 `rg` 仅命中 `tests/base/test_entity_facade.py` 与
  `tests/envs/mdp/test_uniform_velocity_command.py` 的显式测试调用。
- 守护测试：`tests/architecture/test_host_rotation_sync_boundary.py`。
- 迁移目标位置：`unisim/src/unisim/math/rotation_numpy.py`、
  `unisim/src/unisim/math/rotation_tensor.py`、`src/unilab/math`（由 issue #2583
  其他 Child 落地）。

## Related Documents

- [Issue #2583](https://github.com/Motphys/UniLab/issues/2583)
- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
