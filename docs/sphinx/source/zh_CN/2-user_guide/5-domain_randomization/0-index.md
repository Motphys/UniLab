# 域随机化


本页仅描述仓库中那些已经注册、且已经接入 DR provider 的任务的当前状态。所有结论都来自代码；不从设计意图推断任何内容。

当前统一的入口点位于 `NpEnv._init_domain_randomization()` 和 `DomainRandomizationManager`：

- init 路径：task provider 产生一个 `InitRandomizationPlan`；manager 在 env 初始化期间调用后端的 `apply_init_randomization(...)`
- reset 路径：task provider 产生一个 `ResetPlan`；manager 验证能力，然后调用后端的 `set_state(..., randomization=...)`
- interval 路径：task provider 产生一个 `IntervalRandomizationPlan`；manager 在 step 之前按需调用后端的 `apply_interval_randomization(...)`

这三条路径对应三个生命周期类别：

- **init 生命周期 DR**：改变模型 identity 或模型几何的项；只能在 env/backend 初始化和 materialization 期间生效，例如 Sharpa 手物体的 `geom_size` 缩放。
- **reset 生命周期 DR**：不改变模型 identity，只在同一模型内改变参数或 reset 状态的项，例如 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`。
- **interval 生命周期 DR**：step 之间的外部扰动，例如 push。

## 状态结论

1. 当前所有接入 DR provider 的任务都使用统一的 DR 入口点；没有任何任务绕开 `DomainRandomizationManager` 在 `reset()` 内部运行单独的 DR 流程。
2. 它们的结构都大致相同：task 文件定义一个 `domain_rand` 配置 dataclass、一个 `DomainRandomizationProvider` 和一个 `ResetPlan`；`G1WalkFlat` 复用 `G1Walk` 的 provider。
3. 今天所"统一"的主要是入口点和执行流程，而不是每一个随机化项本身。共享辅助函数 `build_common_reset_randomization()` 目前生成 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`；共享的 interval 辅助函数目前只生成 push。
4. `ResetRandomizationPayload` 已经可以表达 `gravity`、`body_iquat`、`body_inertia`、`kp`、`kd`，并且 `MuJoCoBackend` 已声明支持。这些是否实际被使用，仍取决于 task provider 是否对它们进行采样和 dispatch。
5. `MotrixBackend` 目前支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd` 和 interval push；并且它要求在初始化期间所有模型 actuator 都是 position actuator。
6. `geom_size` 不是 reset 生命周期字段；Sharpa 手物体的 geom 缩放由 init 生命周期的模型 materialization 处理。

## 统一性评估表

| Task | 使用统一 DR 入口？ | 结构化形式？ | reset 形式 | interval 形式 | Code |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | task 状态采样 + common payload | push | `go1/joystick.py` |
| `Go2JoystickFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | task 状态采样 + common payload | push | `go2/joystick.py` |
| `G1WalkFlat` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | task 状态采样 + common payload | push | `g1/joystick.py` |
| `G1WalkRough` | 是 | 是：复用 `G1WalkDomainRandomizationProvider` | task 状态采样 + common payload | push | `g1/joystick.py` |
| `G1MotionTracking` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 大量 task 专属的 reset 采样 + common payload | push | `motion_tracking/g1/tracking.py` |
| `AllegroInhandRotation` | 是 | 是：`DomainRandConfig + Provider + ResetPlan` | task 专属的 reset 采样 + common payload | 无 | `allegro_inhand/rotation.py` |
| `SharpaInhandRotation` | 是 | 是：`InitRandomizationPlan + ResetPlan + IntervalRandomizationPlan` | grasp cache 采样 + common payload | 物体 `body_force` | `sharpa_inhand/rotation.py` |
| `SharpaInhandRotationGrasp` | 是 | 是：复用 Sharpa rotation provider 并 override reset 采样 | grasp 收集 reset + common payload | 无 | `sharpa_inhand/grasp_gen.py` |

## 各任务域随机化清单

| Task | 当前已实现的 reset 域随机化 | 当前已实现的 interval 域随机化 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | `base_mass_delta`、`base_com_offset` 和 push 默认启用；`gravity` 默认禁用 |
| `Go2JoystickFlat` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认启用）；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | kp/kd 默认启用；common payload 和 push 默认禁用 |
| `G1WalkFlat` | base xy；base yaw；由 `reset_base_qvel_limit` 采样的 base qvel；command 采样；`gait_phase` 采样；`current_actions/last_actions` 清零；kp/kd 随机化（默认启用）；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | kp/kd 默认启用；common payload 和 push 默认禁用 |
| `G1WalkRough` | 与 `G1WalkFlat` 相同，直接复用同一 provider | `push_robots` | kp/kd 默认启用；common payload 和 push 默认禁用 |
| `G1MotionTracking` | 动作帧采样；root 位姿扰动 `x/y/z/roll/pitch/yaw`；root 速度扰动 `x/y/z/roll/pitch/yaw`；关节位置噪声；在 MuJoCo 下被关节范围 clip；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | `push_robots` | `pose_randomization`、`velocity_randomization`、`joint_position_range` 默认有非零扰动；common payload 和 push 默认禁用 |
| `AllegroInhandRotation` | 若存在 grasp cache，则随机采样一个 grasp；否则对手部关节施加 `joint_noise` 并对球施加 `ball_z_offset`；始终对球的线速度施加 `ball_vel_noise`；可选 common reset 随机化 payload（含 `gravity`） | 无 | 若 grasp cache 路径可用则默认采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认为 0；common payload 默认禁用 |
| `SharpaInhandRotation` | grasp cache 按 `scale_ids` 分桶采样；物体位姿 / quat reset；可选 common reset 随机化 payload（含 `gravity`） | 物体 `body_force` 直接力扰动 | `domain_rand.scale_list` 默认值来自 owner YAML；在 MuJoCo 下，物体 geom 缩放在 init 期间 materialize；common payload 默认禁用；物体 force 通过 Sharpa owner YAML 默认启用 |
| `SharpaInhandRotationGrasp` | 手部位姿 reset；物体位姿 / quat reset；收集成功的 grasp 并按 `scale_ids` 分桶存储；可选 `base_mass_delta`；可选 `base_com_offset`；可选 `gravity` | 无 | 默认用于生成 Sharpa grasp cache；cache 文件名包含单个 scale 值；common payload 默认禁用 |

## 当前统一 DR 的能力与边界

### 1. 统一入口点是完整的

统一入口点由 `NpEnv` 和 `DomainRandomizationManager` 保证：

- 任务只需注册一个 provider
- manager 统一执行能力验证
- 后端统一负责实际施加随机化 payload

因此从执行路径的角度看，这些任务已经是统一的。

### 2. 共享辅助函数仍然较窄

`dr_utils.py` 目前只有两类共享辅助函数：

- reset common payload：`base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`
- interval common payload：push

这意味着：

- 尽管运动控制任务都走统一入口点，但它们的 base xy、yaw、qvel、command 和 gait phase 仍然直接在各自的 provider 内部采样
- `G1MotionTracking` 的 pose / velocity / joint 噪声也是 task 专属逻辑
- Allegro 的 grasp / 物体初始状态采样完全是 task 专属逻辑
- Sharpa 的 `geom_size` 缩放是 init 生命周期的模型 materialization，不属于 reset common payload

所以今天的"统一性"更多是关于 contract 和调用约定，而不是"所有任务共享同一套随机化项 schema"。

### 3. 后端能力已经超出任务当前使用的范围

`ResetRandomizationPayload` 现在包含：

- `base_mass_delta`
- `base_com_offset`
- `gravity`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

当前的后端能力：

- `MuJoCoBackend`：支持上述 7 个 reset 项，外加 interval push 和 interval body force
- `MotrixBackend`：支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd`，外加 interval push；要求在初始化期间 actuator 全部为 position actuator

说明：

- 当前的 `IntervalRandomizationPlan` 支持 `push_perturbation_limit`、`body_linear_velocity_delta` 和 `body_force`；其中 `body_force` 表达热路径上的直接外力扰动，而不暴露后端私有的 `xfrc_applied` 细节。
- 当前 MuJoCo 后端的 interval push 和 interval body force 都通过 `xfrc_applied` dispatch；Sharpa 手物体扰动已切换为直接力扰动。
- Motrix 后端目前仍不支持直接 body-force 扰动，因此这类 owner 配置必须继续显式禁用。

但在任务侧，当前的现实是：并非每个 provider 都构造这些字段。后端 contract 是能力边界；task 配置和 provider 是否 dispatch 一个 payload，才决定了某个任务是否实际启用对应的 DR 项。

## Reset gravity 用法

`gravity` 是一个 reset 生命周期 DR：在每次 reset 时，会按 env 子集采样一个完整的 MuJoCo gravity 向量 `(gx, gy, gz)`，并通过 `ResetRandomizationPayload.gravity` dispatch 到后端。该向量同时表达方向和大小：

- 方向：由 `(gx, gy, gz)` 的方向决定。
- 大小：由向量范数 `sqrt(gx^2 + gy^2 + gz^2)` 决定。
- 生命周期：仅在 reset 时采样和写入；env 会保留该重力，直到下一次 reset 重新采样。
- 后端：当前在 UniLab 中，只有 MuJoCo 后端声明支持该 reset 项；Motrix 后端不支持。一些任务按能力过滤并跳过它；另一些任务在 validate 阶段抛出错误。

配置入口在每个任务的 `env.domain_rand` 下：

```yaml
env:
  domain_rand:
    randomize_gravity: true
    gravity_range:
      - [-0.2, -0.2, -10.5]
      - [0.2, 0.2, -8.5]
```

字段语义：

- `randomize_gravity`：是否启用 gravity reset DR；默认为 `false`。
- `gravity_range`：一个形状为 `(2, 3)` 的逐维采样范围；第一行和第二行给出每个分量的上界和下界。
- 在每次 reset 时，每个维度在 `[min(row0, row1), max(row0, row1)]` 内均匀采样。方向不会自动归一化，重力范数也不固定。

如果你只想随机化大小而保持竖直向下的方向，只开放 `z` 分量：

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

如果你想同时随机化方向和大小，开放 `x/y/z`：

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[-0.3,-0.3,-10.5],[0.3,0.3,-8.5]]'
```

说明：

- `gravity_range` 必须可转换为 `(2, 3)` 数组；否则 reset 在构造 payload 时会抛出错误。
- 该项不调用 `mj_setConst`；MuJoCo step / forward 直接读取 `mjModel.opt.gravity`。
- 不要在 Motrix 后端下启用该项；当前 Motrix 能力不包含 `gravity`。
- 如果你当前的环境仍安装了不包含 `gravity` 字段的 `mujoco-uni` 包，MuJoCo reset 会抛出 unsupported field；你需要使用包含该字段的 `mujoco-uni` 构建/发布版本。
- 在训练期间，建议从较小的倾斜范围开始；否则在早期采样到过大的水平重力，可能会使任务退化为不可学习。

## Interval push 用法

支持 interval push 的任务在 `env.domain_rand` 下配置它：

```yaml
env:
  domain_rand:
    push_robots: true
    push_interval: 750
    max_force: [1.0, 1.0, 0.5]
    push_body_name: null
```

- `push_robots`：是否启用 push。
- `push_interval`：每 N 个 env step 触发一次。
- `max_force`：一个长度为 3 的外力上限；每个维度在 `[-max_force, max_force]` 内采样。
- `push_body_name`：施加力的目标 body / link。默认为 `null`，表示使用后端的 `base_name`。

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]' \
  env.domain_rand.push_body_name=torso_link
```

说明：

- MuJoCo 按 body name 解析，Motrix 按 link name 解析；缺失的 name 会在 env/backend 初始化期间抛出错误。
- `push_body_name` 是一个 init 配置；在 env 创建之后修改它不会改变已经解析的目标。
- 热路径只采样和施加外力；它不解析 XML / asset，也不探测后端私有能力。
- MuJoCo push 通过 `xfrc_applied` 外力实现，不直接覆盖 base 速度。

## `geom_size` 生命周期边界

`geom_size` 明确不属于 `ResetRandomizationPayload`，并且不得在热路径上通过 `BatchEnvPool.reset(..., randomization=...)` 修改。

原因在于 `geom_size` 会改变模型几何和模型 identity；正确的生命周期是：

1. task provider 在 `build_init_randomization_plan(...)` 中生成模型变体以及 env 到模型的分配。
2. MuJoCo 后端在冷路径上使用 `MjSpec` 修改 geom size，并编译 scale 专属的 `MjModel`。
3. 后端使用长度为 `num_envs` 的模型序列构造 `BatchEnvPool`。

```{toctree}
:hidden:

1-configuration
2-writing_providers
```
4. reset 阶段只在同一模型 identity 内执行状态和参数扰动；它不处理 `geom_size`。

这条边界存在的目的是遵循冷路径 asset/model-metadata 访问原则：`step()`、`reset()` 和热路径 DR 不解析 XML、不读取 asset，也不在运行时基于 asset 元数据进行分支。

## Sharpa 手物体 Geom 缩放用法

Sharpa 手是仓库中当前 `geom_size` init 生命周期 DR 的示例任务。相关任务配置：

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`

### 1. 配置入口

Sharpa 的 scale 配置位于 env owner YAML 的 `env.domain_rand.scale_list`：

```yaml
env:
  object_body_name: object
  object_geom_name: object
  domain_rand:
    scale_list: [0.5, 0.6, 0.7, 0.8]
```

字段语义：

- `object_body_name`：物体 body name；用于在 reset / observation 期间定位物体 body，而不是缩放的目标字段。
- `object_geom_name`：要缩放的 MuJoCo geom name；默认为 `object`。
- `domain_rand.scale_list`：显式的 scale 列表；每个值必须大于 0。
- `domain_rand.scale_list` 的顺序就是 `scale_id` 的顺序。
- `domain_rand.scale_list` 的长度就是模型变体的数量。

每个 env 被静态分配一个 `scale_id`。当前的分配规则是连续分桶分配；当 `algo.num_envs` 不能被 `num_scales` 整除时，前几个 scale 桶各多分到一个 env：

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

如果 `algo.num_envs=4096` 且 `num_scales=4`，那么每 1024 个 env 使用同一个 scale 桶。

### 2. MuJoCo Materialization 行为

MuJoCo 后端如何应用它：

1. env/provider 在 init 期间基于 `scale_list` 构建 `ModelVariantSpec`。
2. 后端使用 `MjSpec` 读取模型，并修改与 `object_geom_name` 对应的 geom 的 `size`。
3. 每个 scale 编译一个 scale 专属的 `MjModel`。
4. 在首次需要物理 pool 时，将 env 到模型的分配展开为长度为 `num_envs` 的模型序列，然后构造 `BatchEnvPool`。

因此，`domain_rand.scale_list` 仅在 env/backend 初始化期间生效。在 env 创建之后修改 `env.domain_rand.scale_list` 不会改变已经 materialize 的模型 pool。

该流程有三条重要边界：

- `BatchEnvPool` 是惰性构造的；正常路径不会先为默认模型构造一个 pool，再为 `scale_list` 重建它。
- 多个模型变体的编译使用基于进程的并行分块完成；不要在 Python 线程中编译，也不要在上层 for 循环中串行编译 `num_envs` 个模型。
- worker 使用 `MjSpec` 编译变体并保存 `.mjb`；父进程仅按 `.mjb` 路径加载 `MjModel.from_binary_path(...)`。不要通过 IPC 传回修改后的模型对象或模型字节。

### 3. Grasp Cache 与 Scale 桶

Sharpa rotation 任务按 `scale_ids` 从多个单 scale 的 grasp cache 中采样：

- cache 文件名默认由 `grasp_cache_path` 和单个 scale 值共同决定。
- `scale_list: [0.5, 0.6, 0.7, 0.8]` 默认对应 `cache/sharpa_grasp_linspace_0.5.npy`、`cache/sharpa_grasp_linspace_0.6.npy`、`cache/sharpa_grasp_linspace_0.7.npy`、`cache/sharpa_grasp_linspace_0.8.npy`。
- 在 rotation 启动时会检查 `scale_list` 的所有 cache 文件；如果有任何缺失，则报错。
- 每个 scale 桶只从其自身 scale 的 cache 文件中采样，避免在不同物体 scale 之间混用 grasp 初始状态。

在生成多 scale cache 时，分别多次运行 grasp 收集任务：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.5]' algo.num_envs=4096
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.6]' algo.num_envs=4096
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.7]' algo.num_envs=4096
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' algo.num_envs=4096
```

或顺序使用仓库中的辅助脚本：

```bash
./scripts/sharpa_collect_grasps.sh 0.5 0.6 0.7 0.8
```

然后用相同的 `scale_list` 训练 rotation：

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

### 4. 边界与注意事项

- `geom_size` 不是 reset DR 字段，且不得写入 `ResetPlan.randomization`。
- `BatchEnvPool.reset(..., randomization=...)` 目前不支持 `geom_size`。
- `geom_size` 缩放仅在 MuJoCo 后端下 materialize；Motrix 后端目前不会从 `scale_list` 产生多模型 pool。
- `scale_list` 的长度是模型变体的数量，而不是每次 reset 的重采样次数。
- 每个 env 的 `scale_id` 在 init 期间静态分配，且在 reset 时不变。
- 在扩容时，扩容模型变体的数量；不要按 `num_envs` 为每个 env 编译一个模型。多个 env 共享与同一 scale 桶对应的同一个 `MjModel`。
- 热路径不得读取 XML、解析 asset，也不得使用 `getattr` / `hasattr` 探测后端私有能力来决定缩放行为。
- 在扩展到其他 shape DR 时，优先复用 init 生命周期 contract；不要把 shape 字段塞进 reset payload。

有关配置示例，请参阅 {doc}`1-configuration`。有关开发者
provider 接口和后端能力边界，请参阅
{doc}`2-writing_providers` 和 {doc}`Domain Randomization Contract </zh_CN/4-developer_guide/2-contracts/4-dr_contract>`。
