# 域随机化


本页仅描述仓库中已注册任务的域随机化现状。所有结论都来自代码；不从设计意图推断任何内容。

Manager-Based event term 是唯一 DR 声明路径：

- **Manager-Based（Compatible）任务**：reset / interval 随机化通过 owner YAML 中的 Hydra `events:` manager term 声明；reset 生命周期的 event 在 reset 时采样，interval 生命周期的 event 在 step 之间施加扰动。例如 `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 的 `events:` 段。



这三条路径对应三个生命周期类别：

- **construction 生命周期 identity**：固定 model/tool variant 及其 immutable env assignment，只在 backend construction/materialization 期间生效。
- **reset 生命周期 DR**：不改变模型 identity，只在同一模型内改变参数或 reset 状态的项，例如 `base_mass_delta`、`base_com_offset`、`gravity`、`kp`、`kd`，以及 backend 显式声明支持的 geometry/model 字段。
- **interval 生命周期 DR**：step 之间的外部扰动，例如 push。

## 状态结论

1. reset/interval 随机化由 owner YAML 中的 `events:` manager term 声明，并由 manager 生命周期统一执行。
2. Manager-Based owner 通过 Hydra command/event term 声明 reset 行为。G1 motion reset 扰动归 `MotionCommandCfg` 所有，WBT 另加 `EventTermCfg` reset 与 interval term。
3. `ResetRandomizationPayload` 表达 curated reset terms；backend 必须声明每个请求 term，并拥有其派生量重算义务。
4. `MotrixBackend` 目前支持 `base_mass_delta`、`base_com_offset`、`kp`、`kd` 和 interval push；并且它要求在初始化期间所有模型 actuator 都是 position actuator。
5. 固定 mesh/tool identity 由 `env.fixed_model_variants` 声明；reset-time geometry 字段仍位于 backend capability 声明之后，且不会改变该 identity。

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

## 当前 DR 的能力与边界

owner YAML 声明 event term；`ResetStateTransaction` 组合 selected rows 并校验
shape；UniSim backend 声明并应用 curated payload。task-specific reset 采样仍由
command/event term 拥有：

- `G1MotionTracking` 的 pose / velocity / joint noise 归 manager command 所有。
- Allegro grasp / object 初始状态采样是 task-specific event logic。
- 固定 model/tool identity 是 construction-time，不是 reset-time DR。

未显式声明支持的后端能力会 fail closed；不存在过滤或静默回退。

## Reset gravity 用法

`gravity` 是 reset 生命周期 DR：每次 reset 按选中环境采样完整的 MuJoCo gravity
向量 `(gx, gy, gz)`，并通过 `ResetRandomizationPayload.gravity` 提交。请通过调用
`randomize_physics_scene_gravity` 的 reset `EventTermCfg` 配置；不支持的后端
fail closed。建议从较小倾斜范围开始，避免早期训练任务不可学习。

## Interval push 用法

Manager-Based 任务通过 `env.events.push_robot` term 配置周期推扰。例如，
`src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 使用
`push_by_setting_velocity`，间隔为 15 秒，并按轴声明速度范围。

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

## 固定 Model/Tool Variant 边界

Manager-Based owner 在 environment config 中声明 fixed variants。Task 拥有
名称、source descriptor 和最终 assignment，但不编译模型：

```yaml
env:
  fixed_model_variants:
    variants:
      - name: tool_a
        source_model_file: tools/tool_a.xml
      - name: tool_b
        source_model_file: tools/tool_b.xml
    # 省略 explicit_variant_names 时使用确定性 round-robin assignment。
    explicit_variant_names: [tool_a, tool_b]
```

Manager factory 会把它物化为形状 `(num_envs,)`、只读的 `int32` assignment；
空 explicit list 选择 round-robin。Assignment 是
task identity：backend construction 后固定，reset event 不会重新采样。

UniLab 不打开、解析或编译 `source_model_file`，也不持有 `MjSpec`、`MjModel`、
mjbatch 或 Warp object。UniSim adapter 负责 source realization，并必须声明
`supports_fixed_variants`。在该 contract 落地前，配置 fixed variants 的
task 会在 env 构造前 fail closed。无法投影到统一 public
state/action/sensor layout 的 heterogeneous variants 同样 fail closed。

Reset-time model-field DR 保持在已选 identity 内。其 canonical 或 per-env
基线来自 backend 声明的 `get_reset_term_default(term)` contract；Manager term
不会重新编译场景，也不会把 canonical 基线套到每个工具上。只随机化模型表的一
部分时，未写入的列保留所选环境的 variant 基线。

所有权边界以及 MJWarp/CPU executor 分工记录在
{doc}`ADR-0010 </adr/ADR-0010-fixed-model-variant-ownership-boundary>`。

```{toctree}
:hidden:

1-configuration
```
## 相关任务

- {doc}`G1 Motion Tracking <../4-tasks/2-motion_tracking>`：开启 DR 前先确认 motion 资产和 replay。
- {doc}`Go2 Rough Terrain <../4-tasks/1-locomotion>`：常见的是 mass、COM、friction、push。

有关后端能力边界，请参阅
{doc}`Domain Randomization Contract </zh_CN/4-developer_guide/2-contracts/4-dr_contract>`。
