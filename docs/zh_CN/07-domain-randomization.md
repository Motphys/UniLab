# 域随机化现状

这页只描述当前仓库里已经注册、且已经接入 DR provider 的任务现状。结论全部来自代码，不按设计意图推断。

当前统一入口在 [`NpEnv._init_domain_randomization()`](../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../src/unilab/dr/manager.py)：

- reset 路径：task provider 产出 `ResetPlan`，manager 校验 capability 后调用 backend `set_state(..., randomization=...)`
- interval 路径：task provider 产出 `IntervalRandomizationPlan`，manager 在 step 前按需调用 backend `apply_interval_randomization(...)`

## 现状结论

1. 当前 7 个已注册任务全部使用统一 DR 入口，没有任务绕开 `DomainRandomizationManager` 直接在 `reset()` 里做另一套 DR 流程。
2. 形式上基本都是结构化的：任务文件内定义 `domain_rand` 配置 dataclass、`DomainRandomizationProvider`、`ResetPlan`，`G1WalkTaskMjSAC` 复用 `G1Joystick` 的 provider。
3. 现在“统一”的主要是入口和执行流程，不是所有随机项本身。公共 helper [`build_common_reset_randomization()`](../../src/unilab/dr/dr_utils.py) 目前只生成 `base_mass_delta` 和 `base_com_offset`；公共 interval helper 目前只生成 push。
4. [`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 已经能表达 `body_iquat`、`body_inertia`、`kp`、`kd`，且 [`MuJoCoBackend`](../../src/unilab/base/backend/mujoco_backend.py) 已声明支持；但当前没有任何任务 provider 真正采样并下发这些项。对任务侧来说，它们还没有形成统一可配置的 DR 项。
5. [`MotrixBackend`](../../src/unilab/base/backend/motrix_backend.py) 当前只支持 `base_mass_delta`、`base_com_offset` 和 interval push。

## 统一性评估表

| 任务 | 是否使用统一 DR 入口 | 是否为结构化形式 | reset 形式 | interval 形式 | 代码 |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go1/joystick.py`](../../src/unilab/envs/locomotion/go1/joystick.py) |
| `Go2JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go2/joystick.py`](../../src/unilab/envs/locomotion/go2/joystick.py) |
| `G1JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`g1/joystick.py`](../../src/unilab/envs/locomotion/g1/joystick.py) |
| `G1WalkTaskMjSAC` | 是 | 是：复用 [`G1JoystickDomainRandomizationProvider`](../../src/unilab/envs/locomotion/g1/joystick.py) | 任务状态采样 + common payload | push | [`g1/joystick_sac.py`](../../src/unilab/envs/locomotion/g1/joystick_sac.py) |
| `G1MotionTracking` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 大量任务特有 reset 采样 + common payload | push | [`motion_tracking/g1/tracking.py`](../../src/unilab/envs/motion_tracking/g1/tracking.py) |
| `AllegroInhandRotation` | 是 | 是：`DomainRandConfig + Provider + ResetPlan` | 纯任务特有 reset 采样，`randomization=None` | 无 | [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py) |

## 任务域随机化清单

| 任务 | 当前实现的 reset 域随机 | 当前实现的 interval 域随机 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlatTerrain` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认开启 `base_mass_delta`、`base_com_offset`、push |
| `Go2JoystickFlatTerrain` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认全部关闭 |
| `G1JoystickFlatTerrain` | base xy；base yaw；按 `reset_base_qvel_limit` 采样 base qvel；command 采样；`gait_phase` 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认 common payload 和 push 关闭 |
| `G1WalkTaskMjSAC` | 与 `G1JoystickFlatTerrain` 相同，直接复用同一个 provider | `push_robots` | 默认 common payload 和 push 关闭 |
| `G1MotionTracking` | motion frame 采样；root pose 扰动 `x/y/z/roll/pitch/yaw`；root velocity 扰动 `x/y/z/roll/pitch/yaw`；joint position noise；MuJoCo 下按 joint range clip；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | `pose_randomization`、`velocity_randomization`、`joint_position_range` 默认有非零扰动；common payload 和 push 默认关闭 |
| `AllegroInhandRotation` | 若有 grasp cache 则随机采样 grasp；否则对 hand joints 加 `joint_noise`、对球加 `ball_z_offset`；始终对球线速度加 `ball_vel_noise`；不下发 backend randomization payload | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0 |
| `AllegroInhandRotationSac` | 与 `AllegroInhandRotation` 相同：grasp cache 采样或 hand joint noise、ball z offset、ball velocity noise；不下发 backend randomization payload | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0 |

## 当前统一 DR 能力与边界

### 1. 统一入口已经完成

统一入口由 [`NpEnv`](../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../src/unilab/dr/manager.py) 保证：

- 任务只需要注册 provider
- manager 统一做 capability 校验
- backend 统一负责真正落地 randomization payload

所以从执行路径看，当前各任务已经统一。

### 2. 公共 helper 还比较窄

[`dr_utils.py`](../../src/unilab/dr/dr_utils.py) 当前只有两类公共 helper：

- reset common payload：`base_mass_delta`、`base_com_offset`
- interval common payload：push

这意味着：

- locomotion 任务虽然都走统一入口，但它们的 base xy、yaw、qvel、command、gait phase 仍然是各自 provider 里直接采样
- `G1MotionTracking` 的 pose / velocity / joint noise 也是任务特有逻辑
- Allegro 的 grasp / object 初始状态采样完全是任务特有逻辑

所以“统一形式”目前更多体现在 contract 和调用方式，而不是“所有任务都共用同一套随机项 schema”。

### 3. backend 能力已经超过当前任务使用范围

[`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 现在包含：

- `base_mass_delta`
- `base_com_offset`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

backend capability 当前是：

- [`MuJoCoBackend`](../../src/unilab/base/backend/mujoco_backend.py)：支持上面 6 个 reset term，且支持 interval push
- [`MotrixBackend`](../../src/unilab/base/backend/motrix_backend.py)：只支持 `base_mass_delta`、`base_com_offset`，且支持 interval push

但任务侧当前实际情况是：

- 没有任何 provider 构造 `body_iquat`
- 没有任何 provider 构造 `body_inertia`
- 没有任何 provider 构造 `kp`
- 没有任何 provider 构造 `kd`

也就是说，backend contract 已经先扩了，但任务配置层和 provider 层还没有统一迁移到这些项。

### 4. `kp/kd` 目前仍不是任务 reset DR 的统一项

虽然 MuJoCo backend 已支持 `kp/kd` reset payload，但当前任务里并没有把它们接到 `domain_rand` 配置和 provider 采样逻辑里。

现有 locomotion / Allegro 任务里，控制增益仍然主要是在环境初始化阶段通过 `create_backend(..., position_actuator_gains=...)` 设置，而不是在每次 reset 时通过 provider 下发：

- [`go1/joystick.py`](../../src/unilab/envs/locomotion/go1/joystick.py)
- [`go2/joystick.py`](../../src/unilab/envs/locomotion/go2/joystick.py)
- [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py)

因此从“当前任务 DR 状态”角度，`kp/kd` 还不能算各任务已经统一接入的域随机项。

## MuJoCo `BatchEnvPool` 随机场字段接口现状

这部分只描述 `python/mujoco/batch_env.py` / `python/mujoco/batch_env.cc` 当前已经暴露的接口，不推断未来设计。

### 1. 当前支持的字段

当前 `SUPPORTED_FIELDS` 为：

- `body_mass`
- `body_ipos`
- `body_iquat`
- `body_inertia`
- `dof_armature`
- `geom_friction`
- `kp`
- `kd`

### 2. 当前整块替换方式

当前整块替换入口仍然是：

- `BatchEnvPool.reset(env_ids, initial_state, randomization=...)`

其中：

- `env_ids` 指定这次 reset 要落到哪些 env
- `randomization` 是 `dict[str, ndarray]`
- key 必须属于上面的 `SUPPORTED_FIELDS`
- value 的首维必须等于 `len(env_ids)`
- value 的其余元素个数必须等于该字段在单个 `mjModel` 里的整块大小

当前字段大小按底层实现是：

- `body_mass`: `nbody`
- `body_ipos`: `3 * nbody`
- `body_iquat`: `4 * nbody`
- `body_inertia`: `3 * nbody`
- `dof_armature`: `nv`
- `geom_friction`: `3 * ngeom`
- `kp`: `nu`
- `kd`: `nu`

对应到传参形状，可以理解为：

- `body_mass`: `(len(env_ids), nbody)`
- `body_ipos`: `(len(env_ids), 3 * nbody)`
- `body_iquat`: `(len(env_ids), 4 * nbody)`
- `body_inertia`: `(len(env_ids), 3 * nbody)`
- `dof_armature`: `(len(env_ids), nv)`
- `geom_friction`: `(len(env_ids), 3 * ngeom)`
- `kp`: `(len(env_ids), nu)`
- `kd`: `(len(env_ids), nu)`

整块接口的语义仍然是“按 env 子集、按字段整块替换”。

### 3. 当前读取方式

当前读取入口分成两类：

- 整块读取：`pool.get_field(env_id, name) -> np.ndarray`
- 索引读取：`pool.get_field_indexed(env_id, name, indices)`

其中索引读取当前支持：

- `indices` 为单个 `int`
- `indices` 为 `Sequence[int]`

返回语义当前是稳定的：

- `body_ipos` / `body_inertia` / `geom_friction`
  - 单索引返回 `(3,)`
  - 多索引返回 `(k, 3)`
- `body_iquat`
  - 单索引返回 `(4,)`
  - 多索引返回 `(k, 4)`
- `body_mass` / `dof_armature` / `kp` / `kd`
  - 单索引返回标量
  - 多索引返回 `(k,)`

### 4. 当前局部写入方式

当前局部写入入口是：

- `pool.set_field_indexed(env_id, name, indices, value)`

其中：

- `env_id` 只作用于一个目标 env
- `name` 必须属于 `SUPPORTED_FIELDS`
- `indices` 支持单个 `int` 或 `Sequence[int]`
- `value` 的 shape 必须和字段分量语义匹配

当前 setter 语义是：

- `body_ipos` / `body_inertia` / `geom_friction`
  - 单索引写入要求 `value.shape == (3,)`
  - 多索引写入要求 `value.shape == (k, 3)`
- `body_iquat`
  - 单索引写入要求 `value.shape == (4,)`
  - 多索引写入要求 `value.shape == (k, 4)`
- `body_mass` / `dof_armature` / `kp` / `kd`
  - 单索引写入要求 `value` 为标量
  - 多索引写入要求 `value.shape == (k,)`

因此现在有两种使用方式：

- 如果要一次替换一个字段在多个 env 上的整块值，继续用 `reset(..., randomization=...)`
- 如果只想改单个 env 内的某几个 body / geom / dof / actuator 条目，直接用 `get_field_indexed` / `set_field_indexed`

这意味着像“只改某个 geom 的 friction”这类场景，已经不需要“先整块读出 flat payload、手工切片、再整块回写”的上层样板代码

另外，当前底层对 refresh 的处理也已经固定：

- `body_mass`、`body_ipos`、`body_iquat`、`body_inertia`、`dof_armature` 会触发 `mj_setConst` refresh
- `geom_friction`、`kp`、`kd` 不触发 refresh

因此，当前 MuJoCo `BatchEnvPool` 的 reset-lifecycle randomization 接口可以概括为：

- 支持字段是固定白名单
- 读取方式同时支持整块读取和索引读取
- 写入方式同时支持整块替换和单 env 内的索引级局部写入
- 当前索引级接口按字段分量宽度返回稳定 shape，并且在需要时自动做 `mj_setConst` refresh

## 新任务接入时的最低标准

如果要保持和当前代码风格一致，新任务至少应满足：

1. 在任务文件里定义自己的 DR config dataclass
2. 在任务文件里定义 `DomainRandomizationProvider`
3. reset 通过 `ResetPlan` 返回 `qpos`、`qvel`、`info_updates`、`randomization`
4. 如需 interval 扰动，通过 `IntervalRandomizationPlan`
5. 在 env 构造函数里调用 `self._init_domain_randomization(...)`

如果某个随机项要做成“统一 DR 项”，还需要同时满足三层一致：

1. [`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 里有明确字段
2. backend capability 明确声明支持，并在 backend 内真正落地
3. 任务 config / provider 真正采样并下发该字段

缺任何一层，都只能算“底层有能力”或“任务里自己做了随机”，还不能算仓库层面的统一 DR 项。
