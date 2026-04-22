# 域随机化现状

这页只描述当前仓库里已经注册、且已经接入 DR provider 的任务现状。结论全部来自代码，不按设计意图推断。

当前统一入口在 [`NpEnv._init_domain_randomization()`](../../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../../src/unilab/dr/manager.py)：

- init 路径：task provider 产出 `InitRandomizationPlan`，manager 在 env 初始化阶段调用 backend `apply_init_randomization(...)`
- reset 路径：task provider 产出 `ResetPlan`，manager 校验 capability 后调用 backend `set_state(..., randomization=...)`
- interval 路径：task provider 产出 `IntervalRandomizationPlan`，manager 在 step 前按需调用 backend `apply_interval_randomization(...)`

三条路径对应三类生命周期：

- **init-lifecycle DR**：改变模型身份或模型几何的项，只能在 env/backend 初始化与 materialization 阶段生效，例如 Sharpa-hand 的 object `geom_size` 缩放。
- **reset-lifecycle DR**：不改变模型身份、只改变同一个模型内参数或 reset 状态的项，例如 `base_mass_delta`、`base_com_offset`、`kp`、`kd`。
- **interval-lifecycle DR**：step 间的外部扰动，例如 push。

## 现状结论

1. 当前已接入 DR provider 的任务全部使用统一 DR 入口，没有任务绕开 `DomainRandomizationManager` 直接在 `reset()` 里做另一套 DR 流程。
2. 形式上基本都是结构化的：任务文件内定义 `domain_rand` 配置 dataclass、`DomainRandomizationProvider`、`ResetPlan`，`G1WalkFlat` 复用 `G1Joystick` 的 provider。
3. 现在“统一”的主要是入口和执行流程，不是所有随机项本身。公共 helper [`build_common_reset_randomization()`](../../../src/unilab/dr/dr_utils.py) 目前生成 `base_mass_delta`、`base_com_offset`、`kp`、`kd`；公共 interval helper 目前只生成 push。
4. [`ResetRandomizationPayload`](../../../src/unilab/dr/types.py) 已经能表达 `body_iquat`、`body_inertia`、`kp`、`kd`，且 [`MuJoCoBackend`](../../../src/unilab/base/backend/mujoco_backend.py) 已声明支持。是否真正使用这些项，仍取决于任务 provider 是否采样并下发。
5. [`MotrixBackend`](../../../src/unilab/base/backend/motrix_backend.py) 当前支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd` 和 interval push；并在初始化阶段要求模型 actuator 全部为 position actuator。
6. `geom_size` 不属于 reset-lifecycle 字段；Sharpa-hand object geom scale 通过 init-lifecycle 的 model materialization 完成。

## 统一性评估表

| 任务 | 是否使用统一 DR 入口 | 是否为结构化形式 | reset 形式 | interval 形式 | 代码 |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go1/joystick.py`](../../../src/unilab/envs/locomotion/go1/joystick.py) |
| `Go2JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go2/joystick.py`](../../../src/unilab/envs/locomotion/go2/joystick.py) |
| `G1JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`g1/joystick.py`](../../../src/unilab/envs/locomotion/g1/joystick.py) |
| `G1WalkFlat` | 是 | 是：复用 [`G1JoystickDomainRandomizationProvider`](../../../src/unilab/envs/locomotion/g1/joystick.py) | 任务状态采样 + common payload | push | [`g1/joystick_sac.py`](../../../src/unilab/envs/locomotion/g1/joystick_sac.py) |
| `G1MotionTracking` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 大量任务特有 reset 采样 + common payload | push | [`motion_tracking/g1/tracking.py`](../../../src/unilab/envs/motion_tracking/g1/tracking.py) |
| `AllegroInhandRotation` | 是 | 是：`DomainRandConfig + Provider + ResetPlan` | 纯任务特有 reset 采样，`randomization=None` | 无 | [`inhand_rot_allegro/rotation.py`](../../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py) |
| `SharpaInhandRotation` | 是 | 是：`InitRandomizationPlan + ResetPlan` | grasp cache 采样 + common payload | 无 | [`sharpa_inhand/rotation.py`](../../../src/unilab/envs/manipulation/sharpa_inhand/rotation.py) |
| `SharpaInhandRotationGrasp` | 是 | 是：复用 Sharpa rotation provider 并覆盖 reset 采样 | grasp collection reset + common payload | 无 | [`sharpa_inhand/grasp_gen.py`](../../../src/unilab/envs/manipulation/sharpa_inhand/grasp_gen.py) |

## 任务域随机化清单

| 任务 | 当前实现的 reset 域随机 | 当前实现的 interval 域随机 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认开启 `base_mass_delta`、`base_com_offset`、push |
| `Go2JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认开启）；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1JoystickFlat` | base xy；base yaw；按 `reset_base_qvel_limit` 采样 base qvel；command 采样；`gait_phase` 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认开启）；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1WalkFlat` | 与 `G1JoystickFlat` 相同，直接复用同一个 provider | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1MotionTracking` | motion frame 采样；root pose 扰动 `x/y/z/roll/pitch/yaw`；root velocity 扰动 `x/y/z/roll/pitch/yaw`；joint position noise；MuJoCo 下按 joint range clip；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | `pose_randomization`、`velocity_randomization`、`joint_position_range` 默认有非零扰动；common payload 和 push 默认关闭 |
| `AllegroInhandRotation` | 若有 grasp cache 则随机采样 grasp；否则对 hand joints 加 `joint_noise`、对球加 `ball_z_offset`；始终对球线速度加 `ball_vel_noise`；下发 common reset randomization payload | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0 |

| `SharpaInhandRotation` | grasp cache 按 `scale_ids` 分桶采样；object pose / quat reset | 无 | `scale_range` 默认 `[0.5, 0.5, 1]`，MuJoCo 下会在 init 阶段 materialize object geom scale |
| `SharpaInhandRotationGrasp` | hand pose reset；object pose / quat reset；采集成功 grasp 并按 `scale_ids` 分桶保存；可选 `base_mass_delta`；可选 `base_com_offset` | 无 | 默认用于生成 Sharpa grasp cache，cache 文件名包含 `scale_range` tag |

## 当前统一 DR 能力与边界

### 1. 统一入口已经完成

统一入口由 [`NpEnv`](../../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../../src/unilab/dr/manager.py) 保证：

- 任务只需要注册 provider
- manager 统一做 capability 校验
- backend 统一负责真正落地 randomization payload

所以从执行路径看，当前各任务已经统一。

### 2. 公共 helper 还比较窄

[`dr_utils.py`](../../../src/unilab/dr/dr_utils.py) 当前只有两类公共 helper：

- reset common payload：`base_mass_delta`、`base_com_offset`、`kp`、`kd`
- interval common payload：push

这意味着：

- locomotion 任务虽然都走统一入口，但它们的 base xy、yaw、qvel、command、gait phase 仍然是各自 provider 里直接采样
- `G1MotionTracking` 的 pose / velocity / joint noise 也是任务特有逻辑
- Allegro 的 grasp / object 初始状态采样完全是任务特有逻辑
- Sharpa 的 `geom_size` scale 是 init-lifecycle model materialization，不属于 reset common payload

所以“统一形式”目前更多体现在 contract 和调用方式，而不是“所有任务都共用同一套随机项 schema”。

### 3. backend 能力已经超过当前任务使用范围

[`ResetRandomizationPayload`](../../../src/unilab/dr/types.py) 现在包含：

- `base_mass_delta`
- `base_com_offset`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

backend capability 当前是：

- [`MuJoCoBackend`](../../../src/unilab/base/backend/mujoco_backend.py)：支持上面 6 个 reset term，且支持 interval push
- [`MotrixBackend`](../../../src/unilab/base/backend/motrix_backend.py)：支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd`，且支持 interval push；初始化阶段要求 actuator 全为 position

但任务侧当前实际情况是：并不是所有 provider 都构造这些字段。backend contract 是能力边界，任务配置和 provider 是否下发 payload 才决定该任务是否实际启用对应 DR 项。

### 4. `geom_size` 的生命周期边界

`geom_size` 明确不属于 [`ResetRandomizationPayload`](../../../src/unilab/dr/types.py)，也不应通过 `BatchEnvPool.reset(..., randomization=...)` 在热路径中修改。

原因是 `geom_size` 会改变模型几何和模型身份，正确生命周期是：

1. task provider 在 `build_init_randomization_plan(...)` 里生成 model variant 和 env-to-model assignment。
2. MuJoCo backend 在冷路径用 `MjSpec` 修改 geom size，编译 scale-specific `MjModel`。
3. backend 用长度为 `num_envs` 的 model sequence 构造 `BatchEnvPool`。
4. reset 阶段只做同一 model identity 内的状态和参数扰动，不处理 `geom_size`。

这个边界是为了遵守冷路径 asset/model metadata 访问原则：`step()`、`reset()` 和热路径 DR 不解析 XML、不读取 asset、不根据 asset 元数据做运行时分支。

## Sharpa-hand object geom scale 用法

Sharpa-hand 是当前仓库里 `geom_size` init-lifecycle DR 的示例任务。相关任务配置为：

- `task=sharpa_inhand/mujoco`
- `task=sharpa_inhand_grasp/mujoco`

### 1. 配置入口

Sharpa 的缩放配置位于 env owner YAML 的 `env.scale_range`：

```yaml
env:
  object_body_name: object
  object_geom_name: object
  scale_range: [0.5, 0.8, 4]
```

字段语义：

- `object_body_name`：object body 名称，用于 reset / observation 中定位 object body，不是 scale 的目标字段。
- `object_geom_name`：要缩放的 MuJoCo geom 名称，默认是 `object`。
- `scale_range[0]`：最小 scale，必须大于 0。
- `scale_range[1]`：最大 scale，必须大于 0。
- `scale_range[2]`：scale 桶数量，必须是正整数。

实际 scale 值由 `np.linspace(lower, upper, num_scales)` 生成。例如：

```yaml
env:
  scale_range: [0.5, 0.8, 4]
```

会生成 4 个 scale：

- `0.5`
- `0.6`
- `0.7`
- `0.8`

每个 env 会被静态分配一个 `scale_id`。当前分配规则是按 bucket 连续分配，因此 `algo.num_envs` 必须能被 `num_scales` 整除：

```bash
uv run python scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.scale_range=[0.5,0.8,4]' algo.num_envs=4096
```

如果 `algo.num_envs=4096`、`num_scales=4`，则每 1024 个 env 使用同一个 scale bucket。

### 2. MuJoCo materialization 行为

MuJoCo backend 的落地方式是：

1. env/provider 在 init 阶段根据 `scale_range` 构造 `ModelVariantSpec`。
2. backend 用 `MjSpec` 读取模型并修改 `object_geom_name` 对应 geom 的 `size`。
3. 每个 scale 编译一套 scale-specific `MjModel`。
4. 第一次需要 physics pool 时，用 env-to-model assignment 展开成长度为 `num_envs` 的 model sequence，再构造 `BatchEnvPool`。

因此，`scale_range` 只在 env/backend 初始化阶段生效。env 创建后再改 `env.scale_range` 不会改变已经 materialize 的模型池。

这个流程有三个重要边界：

- `BatchEnvPool` 是 lazy 构造的；正常路径不会先为默认模型构造一套 pool，再为了 `scale_range` 重建一套 pool。
- 多个 model variant 的编译使用 process-based parallelism 分块执行；不要在 Python thread 里编译，也不要在上层 for 循环串行编译 `num_envs` 个模型。
- worker 用 `MjSpec` 编译 variant 并保存 `.mjb`，父进程只按 `.mjb` 路径加载 `MjModel.from_binary_path(...)`；不要通过 IPC 回传修改后的模型对象或模型 bytes。

### 3. grasp cache 与 scale bucket

Sharpa rotation 任务按 `scale_ids` 从 grasp cache 里分桶采样：

- cache 文件名默认由 `grasp_cache_path` 和 `scale_range` 共同决定。
- `scale_range: [0.5, 0.8, 4]` 默认对应类似 `cache/sharpa_grasp_linspace_0.5-0.8-4.npy` 的路径。
- cache 行数必须能被 `num_scales` 整除。
- 每个 scale bucket 使用自己的 cache 区间，避免不同 object scale 混用 grasp 初始状态。

生成多 scale cache 时，应使用同一套 `scale_range` 跑 grasp 采集任务：

```bash
uv run python scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.scale_range=[0.5,0.8,4]' algo.num_envs=4096
```

随后训练 rotation 时使用相同的 `scale_range`：

```bash
uv run python scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.scale_range=[0.5,0.8,4]' algo.num_envs=4096
```

### 4. 边界和注意事项

- `geom_size` 不是 reset DR 字段，不能写进 `ResetPlan.randomization`。
- `BatchEnvPool.reset(..., randomization=...)` 当前不支持 `geom_size`。
- `geom_size` scale 只在 MuJoCo backend 下 materialize；Motrix backend 当前不会按 `scale_range` 生成多模型池。
- `scale_range[2]` 是模型 variant 数量，不是每次 reset 随机抽样次数。
- 每个 env 的 `scale_id` 是 init 阶段静态 assignment，不会在 reset 时变化。
- 扩 scale 时应扩 model variant 数量，不应按 `num_envs` 编译一 env 一模型；多个 env 共享同一个 scale bucket 对应的 `MjModel`。
- 热路径不得读取 XML、解析 asset 或用 `getattr` / `hasattr` 探测 backend 私有能力来决定 scale 行为。
- 如需扩展到其他 shape DR，应优先复用 init-lifecycle contract，而不是把 shape 字段塞进 reset payload。

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

注意：即使使用 `mujoco-uni==3.6.0.post6`，`geom_size` 仍然不在 reset randomization 的 `SUPPORTED_FIELDS` 里；它在 UniLab 中通过 init-lifecycle model materialization 表达。

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

1. [`ResetRandomizationPayload`](../../../src/unilab/dr/types.py) 里有明确字段
2. backend capability 明确声明支持，并在 backend 内真正落地
3. 任务 config / provider 真正采样并下发该字段

缺任何一层，都只能算“底层有能力”或“任务里自己做了随机”，还不能算仓库层面的统一 DR 项。

## Navigation

- Previous: [G1 Motion Tracking](05-motion-tracking.md)
