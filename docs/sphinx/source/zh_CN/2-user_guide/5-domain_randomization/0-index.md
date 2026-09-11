# 域随机化


本页仅描述仓库中已注册任务的域随机化现状。所有结论都来自代码；不从设计意图推断任何内容。

当前存在两条 DR 声明路径：

- **Manager-Based（Compatible）任务**：reset / interval 随机化通过 owner YAML 中的 Hydra `events:` manager term 声明；reset 生命周期的 event 在 reset 时采样，interval 生命周期的 event 在 step 之间施加扰动。例如 `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 的 `events:` 段。
- **任务级 provider 路径**：自定义任务（包括托管在外部仓库中的任务）可以通过 `DomainRandomizationProvider` + `DomainRandomizationManager` 声明 `env.domain_rand.*` 配置。当前仓库内没有任务使用该路径。

legacy provider 路径的统一入口点位于 `NpEnv._init_domain_randomization()` 和 `DomainRandomizationManager`：

- init 路径：task provider 产生一个 `InitRandomizationPlan`；manager 在 env 初始化期间调用后端的 `apply_init_randomization(...)`
- reset 路径：task provider 产生一个 `ResetPlan`；manager 验证能力，然后调用后端的 `set_state(..., randomization=...)`
- interval 路径：task provider 产生一个 `IntervalRandomizationPlan`；manager 在 step 之前按需调用后端的 `apply_interval_randomization(...)`

这三条路径对应三个生命周期类别：

- **init 生命周期 DR**：改变模型 identity 或模型几何的项；只能在 env/backend 初始化和 materialization 期间生效，例如通过模型变体进行的物体 `geom_size` 缩放。
- **reset 生命周期 DR**：不改变模型 identity，只在同一模型内改变参数或 reset 状态的项，例如 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`。
- **interval 生命周期 DR**：step 之间的外部扰动，例如 push。

## 状态结论

1. Manager-Based 任务不注册 DR provider；它们的 reset/interval 随机化是 owner YAML 中的 `events:` manager term，由 manager 生命周期统一执行。走 provider 路径的自定义任务则经过 `DomainRandomizationManager` 统一入口。
2. provider 路径的 owner 定义 `domain_rand` 配置 dataclass、`DomainRandomizationProvider` 和 `ResetPlan`；Manager-Based owner 则通过 Hydra command/event term 声明 reset 行为。G1 motion reset 扰动归 `MotionCommandCfg` 所有，WBT 另加 `EventTermCfg` reset 与 interval term。
3. 今天所"统一"的主要是入口点和执行流程，而不是每一个随机化项本身。legacy 路径的共享辅助函数 `build_common_reset_randomization()` 目前生成 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`。
4. `ResetRandomizationPayload` 已经可以表达 `gravity`、`body_iquat`、`body_inertia`、`kp`、`kd`，并且 `MuJoCoBackend` 已声明支持。这些是否实际被使用，仍取决于 task provider 是否对它们进行采样和 dispatch。
5. `MotrixBackend` 目前支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd` 和 interval push；并且它要求在初始化期间所有模型 actuator 都是 position actuator。
6. `geom_size` 不是 reset 生命周期字段；物体 geom 缩放由 init 生命周期的模型 materialization 处理。

## 统一性评估表

| Task | 声明路径 | 结构化形式？ | reset 形式 | interval 形式 | Code |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlat` | Hydra `events:` term | 是：owner YAML 声明 reset/interval event | root-state reset + base mass/COM + `pd_gains` | `push_by_setting_velocity` event | `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` |
| `Go2JoystickFlat` | Hydra `events:` term | 是：owner YAML 声明 reset event | root-state reset + `pd_gains` kp/kd | 无 | `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml` |
| `G1WalkFlat` | Hydra `events:` term | 是：Hydra `EventTermCfg` + Manager-Based reset term | root-state reset + 经 `pd_gains` 的 kp/kd | 无 | `g1/manager_terms.py` |
| `G1WalkRough` | Hydra `events:` term | 是：与 `G1WalkFlat` 相同的 Manager-Based event term | root-state reset + 经 `pd_gains` 的 kp/kd | 无 | `g1/manager_terms.py` |
| `G1MotionTracking` | Hydra command term | 是：Hydra `MotionCommandCfg` + Manager-Based command reset | motion frame、root pose/velocity 与 joint-position 采样 | 无 | `motion_tracking/common/manager_terms.py` |
| `G1WBTObs` | Hydra `events:` term | 是：同一 motion command + Hydra `EventTermCfg` | motion reset 加 mass/COM/PD/friction/encoder-bias event | interval velocity kick | `motion_tracking/g1/manager_terms.py` |
| `AllegroInhandRotation` | Hydra `events:` term | 是：Hydra `EventTermCfg` + Manager-Based reset term | entity 范围的手/球 reset | 无 | `allegro_inhand/manager_terms.py` |
| `AllegroInhandRotationGrasp` | Hydra `events:` term | 是：复用 rotation reset event + `RecorderTermCfg` | 带噪声的手部 reset + grasp 收集 | 无 | `allegro_inhand/grasp_gen.py` |

## 各任务域随机化清单

| Task | 当前已实现的 reset 域随机化 | 当前已实现的 interval 域随机化 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlat` | 经 `reset_root_state_uniform` 的 base xy/yaw 与 base qvel；command 采样（`UniformVelocityCommandCfg`）；经 `randomize_rigid_body_mass` 的 base mass；经 `randomize_rigid_body_com` 的 base COM；经 `pd_gains` 的 kp/kd | `push_by_setting_velocity` interval event | 上述 event term 全部在 `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 中默认声明并启用 |
| `Go2JoystickFlat` | 经 `reset_root_state_uniform` 的 base xy/yaw 与 base qvel；command 采样；经 `pd_gains` 的 kp/kd | 无 | event term 在 `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml` 中默认声明并启用 |
| `G1WalkFlat` | 经 `reset_root_state_uniform` 的 base xy/yaw 与 base qvel；带平面死区的 command 采样；`gait_phase` 采样；经 `pd_gains` 的 kp/kd 随机化 | 无 | mujoco owner 默认启用 kp/kd；motrix/mjwarp owner 默认禁用 |
| `G1WalkRough` | 与 `G1WalkFlat` 相同（共享 owner base，rough 场景） | 无 | 与 `G1WalkFlat` 相同的默认值 |
| `G1MotionTracking` | Motion-command frame 采样；root 位姿扰动 `x/y/z/roll/pitch/yaw`；root 速度扰动 `x/y/z/roll/pitch/yaw`；通过 public entity soft limit clip 的关节位置噪声；action-manager 状态 reset | 无 | base owner 中 `pose_range`、`velocity_range` 与 `joint_position_range` 默认有非零扰动 |
| `G1WBTObs` | 同一 motion reset 加 base mass、base COM、PD gain、足端摩擦和 encoder-bias event term | `push_by_setting_velocity` | WBT owner 显式启用上述全部 event term；能力不支持时直接报错，不回退 |
| `AllegroInhandRotation` | entity 范围的手/球 reset；显式配置 grasp cache 时进行采样，否则以 `null` 显式选择模型 home pose；可选 `joint_noise`、`ball_velocity_noise` 与 `ball_z_offset` | 无 | owner YAML 显式选择 home pose 与零 reset 噪声；配置的 cache 缺失或格式错误时 fail-closed |
| `AllegroInhandRotationGrasp` | 复用 rotation reset 并设置 `joint_noise=0.25`；Manager-Based termination 检查指尖距离、接触数和球高度；recorder 保存成功 timeout rows | 无 | 生成 5 万行 Allegro grasp cache，成功保存后抛出 `RunComplete` |

## 当前统一 DR 的能力与边界

### 1. legacy provider 入口是统一的

legacy provider 路径的统一入口点由 `NpEnv` 和 `DomainRandomizationManager` 保证：

- 任务只需注册一个 provider
- manager 统一执行能力验证
- 后端统一负责实际施加随机化 payload

因此从执行路径的角度看，provider 路径的任务是统一的；Manager-Based 任务则由 manager 生命周期统一执行 owner YAML 声明的 `events:` term。

### 2. 共享辅助函数仍然较窄

legacy 路径的 `dr_utils.py` 构造并校验通用 reset payload：

- reset common payload：`base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`

这意味着：

- provider 路径的任务直接在各自的 provider 内部采样 task 专属状态
- `G1MotionTracking` 的 pose / velocity / joint 噪声由其 manager command 所有
- Allegro 的 grasp / 物体初始状态采样完全是 task 专属逻辑
- `geom_size` 缩放是 init 生命周期的模型 materialization，不属于 reset common payload

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

- `MuJoCoBackend`：支持上述 7 个 reset 项，外加 interval push、interval body velocity delta（线速度与世界系角速度）和 interval body force/torque
- `MotrixBackend`：支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd`，外加 interval push；要求在初始化期间 actuator 全部为 position actuator

说明：

- 当前的 `IntervalRandomizationPlan` 支持 `push_perturbation_limit`、`body_linear_velocity_delta`、`body_angular_velocity_delta`、`body_force` 和 `body_torque`；其中 `body_force`/`body_torque` 表达热路径上的直接外力/力矩扰动，而不暴露后端私有的 `xfrc_applied` 细节。
- 当前 MuJoCo 后端的 interval push 和 interval body force 都通过 `xfrc_applied` dispatch。
- Motrix 后端目前仍不支持直接 body-force 扰动，因此这类 owner 配置必须继续显式禁用。

但在任务侧，当前的现实是：并非每个 provider 都构造这些字段。后端 contract 是能力边界；task 配置和 provider 是否 dispatch 一个 payload，才决定了某个任务是否实际启用对应的 DR 项。

## Reset gravity 用法

`gravity` 是一个 reset 生命周期 DR：在每次 reset 时，会按 env 子集采样一个完整的 MuJoCo gravity 向量 `(gx, gy, gz)`，并通过 `ResetRandomizationPayload.gravity` dispatch 到后端。该向量同时表达方向和大小：

- 方向：由 `(gx, gy, gz)` 的方向决定。
- 大小：由向量范数 `sqrt(gx^2 + gy^2 + gz^2)` 决定。
- 生命周期：仅在 reset 时采样和写入；env 会保留该重力，直到下一次 reset 重新采样。
- 后端：当前在 UniLab 中，只有 MuJoCo 后端声明支持该 reset 项；Motrix 后端不支持。一些任务按能力过滤并跳过它；另一些任务在 validate 阶段抛出错误。

配置入口位于 provider 路径任务 owner 的 `env.domain_rand` 下；Manager-Based 任务没有 `env.domain_rand`：

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

如果你只想随机化大小而保持竖直向下的方向，只开放 `z` 分量；如果想同时随机化方向和大小，开放 `x/y/z`。在 provider 路径的任务 owner 上，可通过 CLI 以 `env.domain_rand.randomize_gravity=true` 与 `env.domain_rand.gravity_range=[...]` override 启用。

说明：

- `gravity_range` 必须可转换为 `(2, 3)` 数组；否则 reset 在构造 payload 时会抛出错误。
- 该项不调用 `mj_setConst`；MuJoCo step / forward 直接读取 `mjModel.opt.gravity`。
- 不要在 Motrix 后端下启用该项；当前 Motrix 能力不包含 `gravity`。
- MuJoCo 后端通过 `mjbatch` 的 per-simulation 模型展开（`expand("gravity")`）
  写入 gravity，钉住的 `mjbatch` 构建已包含该字段。
- 在训练期间，建议从较小的倾斜范围开始；否则在早期采样到过大的水平重力，可能会使任务退化为不可学习。

## Interval push 用法

Manager-Based 任务通过 `env.events.push_robot` term 配置周期推扰。例如，
`src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 使用
`push_by_setting_velocity`，间隔为 15 秒，并按轴声明速度范围。

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

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

## 相关任务

- {doc}`G1 Motion Tracking <../4-tasks/2-motion_tracking>`：开启 DR 前先确认 motion 资产和 replay。
- {doc}`Go2 Rough Terrain <../4-tasks/1-locomotion>`：常见的是 mass、COM、friction、push。

有关配置示例，请参阅 {doc}`1-configuration`。有关开发者
provider 接口和后端能力边界，请参阅
{doc}`2-writing_providers` 和 {doc}`Domain Randomization Contract </zh_CN/4-developer_guide/2-contracts/4-dr_contract>`。
