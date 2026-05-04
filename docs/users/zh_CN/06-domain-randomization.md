# 域随机化现状

语言: 简体中文

这页只描述当前仓库里已经注册、且已经接入 DR provider 的任务现状。结论全部来自代码，不按设计意图推断。

当前统一入口在 [`NpEnv._init_domain_randomization()`](../../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../../src/unilab/dr/manager.py)：

- init 路径：task provider 产出 `InitRandomizationPlan`，manager 在 env 初始化阶段调用 backend `apply_init_randomization(...)`
- reset 路径：task provider 产出 `ResetPlan`，manager 校验 capability 后调用 backend `set_state(..., randomization=...)`
- interval 路径：task provider 产出 `IntervalRandomizationPlan`，manager 在 step 前按需调用 backend `apply_interval_randomization(...)`

三条路径对应三类生命周期：

- **init-lifecycle DR**：改变模型身份或模型几何的项，只能在 env/backend 初始化与 materialization 阶段生效，例如 Sharpa-hand 的 object `geom_size` 缩放。
- **reset-lifecycle DR**：不改变模型身份、只改变同一个模型内参数或 reset 状态的项，例如 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`。
- **interval-lifecycle DR**：step 间的外部扰动，例如 push。

## 现状结论

1. 当前已接入 DR provider 的任务全部使用统一 DR 入口，没有任务绕开 `DomainRandomizationManager` 直接在 `reset()` 里做另一套 DR 流程。
2. 形式上基本都是结构化的：任务文件内定义 `domain_rand` 配置 dataclass、`DomainRandomizationProvider`、`ResetPlan`，`G1WalkFlat` 复用 `G1Walk` 的 provider。
3. 现在“统一”的主要是入口和执行流程，不是所有随机项本身。公共 helper [`build_common_reset_randomization()`](../../../src/unilab/dr/dr_utils.py) 目前生成 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`；公共 interval helper 目前只生成 push。
4. [`ResetRandomizationPayload`](../../../src/unilab/dr/types.py) 已经能表达 `gravity`、`body_iquat`、`body_inertia`、`kp`、`kd`，且 [`MuJoCoBackend`](../../../src/unilab/base/backend/mujoco_backend.py) 已声明支持。是否真正使用这些项，仍取决于任务 provider 是否采样并下发。
5. [`MotrixBackend`](../../../src/unilab/base/backend/motrix_backend.py) 当前支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd` 和 interval push；并在初始化阶段要求模型 actuator 全部为 position actuator。
6. `geom_size` 不属于 reset-lifecycle 字段；Sharpa-hand object geom scale 通过 init-lifecycle 的 model materialization 完成。

## 统一性评估表

| 任务 | 是否使用统一 DR 入口 | 是否为结构化形式 | reset 形式 | interval 形式 | 代码 |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go1/joystick.py`](../../../src/unilab/envs/locomotion/go1/joystick.py) |
| `Go2JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go2/joystick.py`](../../../src/unilab/envs/locomotion/go2/joystick.py) |
| `G1WalkFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`g1/joystick.py`](../../../src/unilab/envs/locomotion/g1/joystick.py) |
| `G1WalkRough` | 是 | 是：复用 [`G1WalkDomainRandomizationProvider`](../../../src/unilab/envs/locomotion/g1/joystick.py) | 任务状态采样 + common payload | push | [`g1/joystick.py`](../../../src/unilab/envs/locomotion/g1/joystick.py) |
| `G1MotionTracking` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 大量任务特有 reset 采样 + common payload | push | [`motion_tracking/g1/tracking.py`](../../../src/unilab/envs/motion_tracking/g1/tracking.py) |
| `AllegroInhandRotation` | 是 | 是：`DomainRandConfig + Provider + ResetPlan` | 任务特有 reset 采样 + common payload | 无 | [`allegro_inhand/rotation.py`](../../../src/unilab/envs/manipulation/allegro_inhand/rotation.py) |
| `SharpaInhandRotation` | 是 | 是：`InitRandomizationPlan + ResetPlan` | grasp cache 采样 + common payload | 无 | [`sharpa_inhand/rotation.py`](../../../src/unilab/envs/manipulation/sharpa_inhand/rotation.py) |
| `SharpaInhandRotationGrasp` | 是 | 是：复用 Sharpa rotation provider 并覆盖 reset 采样 | grasp collection reset + common payload | 无 | [`sharpa_inhand/grasp_gen.py`](../../../src/unilab/envs/manipulation/sharpa_inhand/grasp_gen.py) |

## 任务域随机化清单

| 任务 | 当前实现的 reset 域随机 | 当前实现的 interval 域随机 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | 默认开启 `base_mass_delta`、`base_com_offset`、push；`gravity` 默认关闭 |
| `Go2JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认开启）；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1WalkFlat` | base xy；base yaw；按 `reset_base_qvel_limit` 采样 base qvel；command 采样；`gait_phase` 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认开启）；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1WalkRough` | 与 `G1WalkFlat` 相同，直接复用同一个 provider | `push_robots` | kp/kd 默认开启；common payload 和 push 默认关闭 |
| `G1MotionTracking` | motion frame 采样；root pose 扰动 `x/y/z/roll/pitch/yaw`；root velocity 扰动 `x/y/z/roll/pitch/yaw`；joint position noise；MuJoCo 下按 joint range clip；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | `pose_randomization`、`velocity_randomization`、`joint_position_range` 默认有非零扰动；common payload 和 push 默认关闭 |
| `AllegroInhandRotation` | 若有 grasp cache 则随机采样 grasp；否则对 hand joints 加 `joint_noise`、对球加 `ball_z_offset`；始终对球线速度加 `ball_vel_noise`；可选 common reset randomization payload（含 `gravity`） | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0；common payload 默认关闭 |
| `SharpaInhandRotation` | grasp cache 按 `scale_ids` 分桶采样；object pose / quat reset；可选 common reset randomization payload（含 `gravity`） | 无 | `scale_list` 默认来自 owner YAML，MuJoCo 下会在 init 阶段 materialize object geom scale；common payload 默认关闭 |
| `SharpaInhandRotationGrasp` | hand pose reset；object pose / quat reset；采集成功 grasp 并按 `scale_ids` 分桶保存；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | 无 | 默认用于生成 Sharpa grasp cache，cache 文件名包含单个 scale 值；common payload 默认关闭 |

## 当前统一 DR 能力与边界

### 1. 统一入口已经完成

统一入口由 [`NpEnv`](../../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../../src/unilab/dr/manager.py) 保证：

- 任务只需要注册 provider
- manager 统一做 capability 校验
- backend 统一负责真正落地 randomization payload

所以从执行路径看，当前各任务已经统一。

### 2. 公共 helper 还比较窄

[`dr_utils.py`](../../../src/unilab/dr/dr_utils.py) 当前只有两类公共 helper：

- reset common payload：`base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`
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
- `gravity`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

backend capability 当前是：

- [`MuJoCoBackend`](../../../src/unilab/base/backend/mujoco_backend.py)：支持上面 7 个 reset term，且支持 interval push 与 interval body force
- [`MotrixBackend`](../../../src/unilab/base/backend/motrix_backend.py)：支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd`，且支持 interval push；初始化阶段要求 actuator 全为 position

注意：

- 当前 `IntervalRandomizationPlan` 支持 `push_perturbation_limit`、`body_linear_velocity_delta` 与 `body_force`；其中 `body_force` 表达热路径直接外力扰动，不暴露 backend 私有 `xfrc_applied` 细节。
- 当前 MuJoCo backend 的 interval push 和 interval body force 都通过 `xfrc_applied` 下发外力；Sharpa-hand 的 object disturbance 已切换为 direct force disturbance。
- Motrix backend 当前仍不支持 direct body-force disturbance，因此此类 owner config 需要继续显式关闭。

但任务侧当前实际情况是：并不是所有 provider 都构造这些字段。backend contract 是能力边界，任务配置和 provider 是否下发 payload 才决定该任务是否实际启用对应 DR 项。

## Reset gravity 用法

`gravity` 是 reset-lifecycle DR：每次 reset 时按 env 子集采样一个完整的 MuJoCo gravity 向量 `(gx, gy, gz)`，并通过 `ResetRandomizationPayload.gravity` 下发给 backend。该向量同时表达方向和大小：

- 方向：由 `(gx, gy, gz)` 的方向决定。
- 大小：由向量范数 `sqrt(gx^2 + gy^2 + gz^2)` 决定。
- 生命周期：只在 reset 时采样和写入；同一个 env 会保持该 gravity，直到下一次 reset 重新采样。
- backend：当前 UniLab 只声明 MuJoCo backend 支持该 reset term；Motrix backend 不支持，部分任务会按 capability 过滤跳过，部分任务会在 validate 阶段报错。

配置入口在各任务的 `env.domain_rand`：

```yaml
env:
  domain_rand:
    randomize_gravity: true
    gravity_range:
      - [-0.2, -0.2, -10.5]
      - [0.2, 0.2, -8.5]
```

字段语义：

- `randomize_gravity`：是否启用 gravity reset DR，默认 `false`。
- `gravity_range`：形状为 `(2, 3)` 的逐维采样区间；第一行和第二行分别给出每个分量的上下界。
- 每次 reset 时会在 `[min(row0, row1), max(row0, row1)]` 内逐维均匀采样，不会自动归一化方向，也不会固定 gravity 模长。

如果只想随机化大小、保持竖直向下方向，可以只放开 `z` 分量：

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

如果想同时随机化方向和大小，可以放开 `x/y/z` 分量：

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[-0.3,-0.3,-10.5],[0.3,0.3,-8.5]]'
```

注意：

- `gravity_range` 必须能转成 `(2, 3)` 数组；否则 reset 构造 payload 时会报错。
- 该项不调用 `mj_setConst`；MuJoCo step / forward 会直接读取 `mjModel.opt.gravity`。
- 不要在 Motrix backend 下开启该项；当前 Motrix capability 不包含 `gravity`。
- 如果当前环境仍安装不包含 `gravity` 字段的 `mujoco-uni` 包，MuJoCo reset 会报 unsupported field；需要使用包含该字段的 `mujoco-uni` 构建/发布版本。
- 训练时建议从小范围倾斜开始，避免一开始采到过大水平重力导致任务退化为不可学习。

## Interval push 用法

支持 interval push 的任务在 `env.domain_rand` 下配置：

```yaml
env:
  domain_rand:
    push_robots: true
    push_interval: 750
    max_force: [1.0, 1.0, 0.5]
    push_body_name: null
```

- `push_robots`：是否启用 push。
- `push_interval`：每隔多少个 env step 触发一次。
- `max_force`：长度为 3 的外力上限；每次在 `[-max_force, max_force]` 内逐维采样。
- `push_body_name`：外力施加目标 body / link。默认 `null`，表示使用 backend 的 `base_name`。

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]' \
  env.domain_rand.push_body_name=torso_link
```

注意：

- MuJoCo 按 body 名解析，Motrix 按 link 名解析；名称不存在会在 env/backend 初始化阶段报错。
- `push_body_name` 是 init 配置，env 创建后再改不会改变已解析的目标。
- 热路径只采样并施加外力，不解析 XML / asset，不做 backend 私有能力探测。
- MuJoCo push 通过 `xfrc_applied` 外力实现，不直接改写 base velocity。

## `geom_size` 的生命周期边界

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

Sharpa 的缩放配置位于 env owner YAML 的 `env.scale_list`：

```yaml
env:
  object_body_name: object
  object_geom_name: object
  scale_list: [0.5, 0.6, 0.7, 0.8]
```

字段语义：

- `object_body_name`：object body 名称，用于 reset / observation 中定位 object body，不是 scale 的目标字段。
- `object_geom_name`：要缩放的 MuJoCo geom 名称，默认是 `object`。
- `scale_list`：显式 scale 列表；每个值都必须大于 0。
- `scale_list` 的顺序就是 `scale_id` 顺序。
- `scale_list` 的长度就是 model variant 数量。

每个 env 会被静态分配一个 `scale_id`。当前分配规则是按 bucket 连续分配，因此 `algo.num_envs` 必须能被 `num_scales` 整除：

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

如果 `algo.num_envs=4096`、`num_scales=4`，则每 1024 个 env 使用同一个 scale bucket。

### 2. MuJoCo materialization 行为

MuJoCo backend 的落地方式是：

1. env/provider 在 init 阶段根据 `scale_list` 构造 `ModelVariantSpec`。
2. backend 用 `MjSpec` 读取模型并修改 `object_geom_name` 对应 geom 的 `size`。
3. 每个 scale 编译一套 scale-specific `MjModel`。
4. 第一次需要 physics pool 时，用 env-to-model assignment 展开成长度为 `num_envs` 的 model sequence，再构造 `BatchEnvPool`。

因此，`scale_list` 只在 env/backend 初始化阶段生效。env 创建后再改 `env.scale_list` 不会改变已经 materialize 的模型池。

这个流程有三个重要边界：

- `BatchEnvPool` 是 lazy 构造的；正常路径不会先为默认模型构造一套 pool，再为了 `scale_list` 重建一套 pool。
- 多个 model variant 的编译使用 process-based parallelism 分块执行；不要在 Python thread 里编译，也不要在上层 for 循环串行编译 `num_envs` 个模型。
- worker 用 `MjSpec` 编译 variant 并保存 `.mjb`，父进程只按 `.mjb` 路径加载 `MjModel.from_binary_path(...)`；不要通过 IPC 回传修改后的模型对象或模型 bytes。

### 3. grasp cache 与 scale bucket

Sharpa rotation 任务按 `scale_ids` 从多份单-scale grasp cache 采样：

- cache 文件名默认由 `grasp_cache_path` 和单个 scale 值共同决定。
- `scale_list: [0.5, 0.6, 0.7, 0.8]` 默认对应 `cache/sharpa_grasp_linspace_0.5.npy`、`cache/sharpa_grasp_linspace_0.6.npy`、`cache/sharpa_grasp_linspace_0.7.npy`、`cache/sharpa_grasp_linspace_0.8.npy`。
- rotation 启动时会检查 `scale_list` 对应的所有 cache 文件是否存在，缺失即报错。
- 每个 scale bucket 只从对应 scale 的 cache 文件采样，避免不同 object scale 混用 grasp 初始状态。

生成多 scale cache 时，应分别跑多次 grasp 采集任务：

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.5]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.6]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.7]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.8]' algo.num_envs=4096
```

也可以顺序执行仓库里的 helper：

```bash
./scripts/sharpa_collect_grasps.sh 0.5 0.6 0.7 0.8
```

随后训练 rotation 时使用相同的 `scale_list`：

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

### 4. 边界和注意事项

- `geom_size` 不是 reset DR 字段，不能写进 `ResetPlan.randomization`。
- `BatchEnvPool.reset(..., randomization=...)` 当前不支持 `geom_size`。
- `geom_size` scale 只在 MuJoCo backend 下 materialize；Motrix backend 当前不会按 `scale_list` 生成多模型池。
- `scale_list` 的长度是模型 variant 数量，不是每次 reset 随机抽样次数。
- 每个 env 的 `scale_id` 是 init 阶段静态 assignment，不会在 reset 时变化。
- 扩 scale 时应扩 model variant 数量，不应按 `num_envs` 编译一 env 一模型；多个 env 共享同一个 scale bucket 对应的 `MjModel`。
- 热路径不得读取 XML、解析 asset 或用 `getattr` / `hasattr` 探测 backend 私有能力来决定 scale 行为。
- 如需扩展到其他 shape DR，应优先复用 init-lifecycle contract，而不是把 shape 字段塞进 reset payload。

底层 `BatchEnvPool` 随机场接口快照和新任务接入标准见 [Domain Randomization Contract](../../developers/zh_CN/domain-randomization-contract.md)。

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [G1 Motion Tracking](05-motion-tracking.md)
