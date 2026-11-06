# 字段参考

本页是 `src/unilab/envs/mdp/events.py` 中域随机化（DR）event term 的字段级
配置参考，与 {doc}`0-index`（审计各注册任务声明了哪些 term）和
{doc}`1-configuration`（说明 owner YAML 的选择方式）互补。下文所有签名、
默认值与取值线索均来自源码或 `src/unilab/conf/` 下的 owner YAML，不从设计
意图推断任何内容。

DR term 运行在三个生命周期类别下（见 {doc}`0-index` 的生命周期概述）：

- **construction**：固定 model/tool identity，在 `env.fixed_model_variants`
  上声明；本页没有任何 event term 属于该类别。
- **reset**：`mode: reset` 的 event term 在每次 episode reset 时采样，并经
  backend 持有的 packed reset transaction 一次性提交。`mode: startup` 是
  reset 生命周期的可选变体：初始化时采样一次，之后回放冻结的逐环境取值。
- **interval**：`mode: interval` 的 event term 在 step 之间按采样计时器施加
  扰动；`mode: step` 每个 step 都触发，由 term 自行管理力的生命周期。

## Backend 适用性

字段 × backend 的精确矩阵由外部包 `unisim-core` 持有
（`DomainRandomizationCapabilities.supported_reset_terms` /
`supported_interval_terms`），无法从本仓库逐字段确认，因此本页刻意不给出
该矩阵。有两条规则始终成立：

1. 所选 backend 未声明支持的 term 在构造期 fail closed；不存在过滤或静默
   回退。
2. Owner YAML 按 backend 分文件
   （`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`）。某个 backend
   不支持的 term 在该 backend 的 owner 中置为 `null`，共享 base owner 为
   其余 backend 保留它。

仓库内的该模式示例：`src/unilab/conf/ppo/task/g1_walk_flat/motrix.yaml` 与
`.../mjwarp.yaml` 把 `pd_gains` 置为 `null`，而 MuJoCo owner 保留 kp/kd
随机化；`src/unilab/conf/sac/task/g1_motion_tracking/genesis.yaml` 把
`foot_friction` 置为 `null`，注释记录了 Genesis 没有逐环境 friction API。
最完整的 owner 实例（mass、COM、gravity、push）是
`src/unilab/conf/flashsac/task/go2_joystick_flat/mujoco.yaml` 的 `events:` 段。

## 通用参数族

下列参数在多个 term 间复用。下方各 term 的表只列出其构造函数实际接受的
参数；未知 param 会在构造期被拒绝。

- `asset_cfg`（`SceneEntityCfg`）：选择 entity 及其组件。`name` 必填（场景
  实体名，例如 `robot`）；组件通过 `body_names`/`body_ids`、
  `joint_names`/`joint_ids`、`geom_names`/`geom_ids` 或
  `actuator_names`/`actuator_ids` 选择。`*_names` 字段接受正则字符串；
  `*_ids` 字段默认为"全部"。多数 term 默认为 `SceneEntityCfg("robot")`
  （整台机器人）。
- `distribution`：`"uniform"`（默认）、`"log_uniform"`（bounds 必须为正）
  或 `"gaussian"`（range 读作 `(mean, std)`，std 非负）。`pd_gains` 只接受
  `uniform`/`log_uniform`。
- `operation`：采样值与 backend 声明的默认值的组合方式：`"add"` →
  `default + sample`，`"scale"` → `default * sample`，`"abs"` → 采样值
  直接覆盖默认值。默认值因 term 而异，见各表。
- range 形状约定：标量对 `[lo, hi]`（如 `kp_range`、
  `mass_distribution_params`）；按轴字典 `{x: [lo, hi], ...}`
  （`com_range`、`pose_range`、`velocity_range`，省略的键默认为 `(0, 0)`）；
  宽度为 3 的 `[[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]`
  （`gravity_distribution_params`）；model-field term 的 `ranges` 还可以是
  作用于所有选中轴的扁平 pair、按轴索引的字典，或 regex → pair 的字典
  （对选中的 geom/joint 名做 `fullmatch`；每个 pattern 至少匹配一个名字且
  不得重叠）。

外层的 term 块（`func`、`mode`、`interval_range_s`、`is_global_time`、
`min_step_count_between_reset`、`params`）是一个 `EventTermCfg`；
`interval_range_s` 与 `is_global_time` 位于 term 上，不在 `params` 内。
`mode: interval` 时 `interval_range_s` 必填；`is_global_time: true` 让所有
环境共用一个计时器。

## Reset 生命周期 Term

### `pd_gains`

同时随机化 actuator 刚度 `kp` 与阻尼 `kd`（`unilab.envs.mdp.pd_gains`）。
仅支持 `mode: reset`；startup 与 interval 在构造期被拒绝。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `kp_range` | `[lo, hi]` | 必填 | 每次 reset 采样 | 逐 actuator 刚度。有限且 `lo <= hi`；`log_uniform` 时必须为正。`operation: scale` 时该 pair 是默认增益的无量纲乘子（G1 owner 用 `[0.9, 1.1]`）；`operation: abs` 时是绝对增益（Go2 owner 用 `kp_range: [31.5, 38.5]`）。 |
| `kd_range` | `[lo, hi]` | 必填 | 每次 reset 采样 | 逐 actuator 阻尼，校验同 `kp_range`。Go2 owner：`kd_range: [0.45, 0.55]`，配合 `abs`。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | 用 `actuator_names`/`actuator_ids` 选择 actuator。 |
| `distribution` | `uniform` \| `log_uniform` | `uniform` | 每次 reset 采样 | 该 term 拒绝 `gaussian`。 |
| `operation` | `scale` \| `abs` | `scale` | 每次 reset 应用 | 该 term 拒绝 `add`。 |

### `randomize_rigid_body_mass`

随机化 body 质量（`unilab.envs.mdp.randomize_rigid_body_mass`）。支持
`mode: reset` 与 `mode: startup`；startup 在初始化时采样一张全宽表并回放。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `asset_cfg` | `SceneEntityCfg` | 必填 | 构造期绑定 | 用 `body_names`/`body_ids` 选择 body。 |
| `mass_distribution_params` | `[lo, hi]` | 必填 | 每次 reset 采样（startup 时一次） | 使用模型的 body 质量单位：本仓库 MuJoCo MJCF 机器人为千克。FlashSAC Go2 MuJoCo owner 在 `base` 上用 `[-1.5, 1.5]` 配合 `operation: add`；SAC G1 Genesis owner 在 startup 下对 `pelvis`/`torso_link` 用 `[0.9, 1.1]` 配合 `operation: scale`。 |
| `operation` | `add` \| `scale` \| `abs` | 必填 | 每次 reset 应用 | 无默认值，必须显式声明。 |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | 每次 reset 采样 | `log_uniform` 要求 bounds 为正。 |
| `recompute_inertia` | bool | `true` | 构造期 | 取 `true` 会抛 `NotImplementedError`（reset payload 没有 inertia 重算契约）；必须显式置 `false`。 |
| `min_mass` | float | `1e-6` | 每次 reset 应用 | 随机化质量的下限截断；必须有限且 `>= 1e-6`。 |

### `randomize_body_mass_inertia`

用一个共享的逐 body 因子同时缩放 body 质量与主惯性矩
（`unilab.envs.mdp.randomize_body_mass_inertia`）。仅支持 `mode: reset`；
因子在首次 reset 时采样一次、按环境缓存，并在之后的 reset 重放，因此行为
等同于 startup term。质心与主轴坐标系保持不变。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `asset_cfg` | `SceneEntityCfg` | 必填 | 构造期绑定 | 用 `body_names`/`body_ids` 选择 body；mass 与 inertia 的绑定必须覆盖同一组 body。 |
| `scale_range` | `[lo, hi]` | 必填 | 首次 reset 时采样一次 | 对数均匀因子 `s = e^{2a}`，其中 `a ~ U(ln(lo)/2, ln(hi)/2)`；bounds 必须为正。质量与主惯性矩都乘以 `s`。 |

### `randomize_rigid_body_com`

加性质心随机化（`unilab.envs.mdp.randomize_rigid_body_com`）。支持
`mode: reset` 与 `mode: startup`。每次应用都会重新读取实时的
`com_range`，因此 curriculum 可以在两次 reset 之间放宽范围。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `com_range` | `{x, y, z: [lo, hi]}` | 必填 | 每次 reset 采样（startup 时一次） | 加到默认 body 质心位置上的偏移，单位为米、在 body 坐标系下。省略的轴默认为 `(0, 0)`。SAC G1 MuJoCo owner 在 `torso_link` 上用 `x: [-0.025, 0.025]`、`y/z: [-0.05, 0.05]`；FlashSAC Go2 owner 在 `base` 上用 `x: [-0.05, 0.05]`。 |
| `asset_cfg` | `SceneEntityCfg` | 必填 | 构造期绑定 | 用 `body_names`/`body_ids` 选择 body。 |

### `randomize_physics_scene_gravity`

按环境随机化完整 gravity 向量
（`unilab.envs.mdp.randomize_physics_scene_gravity`）。仅支持
`mode: reset`。建议从较小倾斜范围开始：过大的水平重力会让早期任务不可
学习。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `gravity_distribution_params` | `[[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]` | 必填 | 每次 reset 采样 | `(gx, gy, gz)` 的逐轴 bounds，单位 m/s^2。`operation: abs` 时采样向量直接覆盖默认值；FlashSAC Go2 MuJoCo owner 声明 `[[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]` 配合 `abs`（零宽度占位，展示形状）。 |
| `operation` | `add` \| `scale` \| `abs` | 必填 | 每次 reset 应用 | 无默认值，必须显式声明。 |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | 每次 reset 采样 | |

### `geom_friction`

随机化三分量 geom friction 三元组（切向、扭转、滚动），与 pin 定的 mjlab
语义一致（`unilab.envs.mdp.geom_friction`）。仅支持 `mode: reset`。最终
取值必须保持非负且有限。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `ranges` | pair / 按轴字典 / regex 字典 | 必填 | 每次 reset 采样 | 扁平 `[lo, hi]` 作用于所有选中轴；`{0: [lo, hi]}` 只针对一个轴；`{".*_foot.*": [lo, hi]}` 按名字用 regex `fullmatch` 分别给范围。SAC G1 MuJoCo owner 在脚部碰撞 geom 上用 `[0.3, 1.2]` 配合 `operation: abs`。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | 用 `geom_names`/`geom_ids` 选择 geom。 |
| `axes` | int 列表 | `(0,)` | 每次 reset 采样 | 合法轴为 `0, 1, 2`；默认只动切向分量。 |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | 每次 reset 采样 | |
| `operation` | `add` \| `scale` \| `abs` | `abs` | 每次 reset 应用 | |
| `shared_random` | bool | `false` | 每次 reset 采样 | 为 `true` 时，bounds 相同的 geom 在每个环境内共享一次采样（SAC G1 MuJoCo owner 使用）。 |

### `dof_armature`

随机化关节 armature —— 用于建模齿轮传动的附加转子惯量
（`unilab.envs.mdp.dof_armature`）。参数族与 `geom_friction` 相同，但字段
为标量：只有轴 `0` 合法。仅支持 `mode: reset`；最终取值必须保持非负且
有限。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `ranges` | pair / `{0: [lo, hi]}` / regex 字典 | 必填 | 每次 reset 采样 | regex 键对选中的 joint 名做 `fullmatch`。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | 用 `joint_names`/`joint_ids` 选择 joint。 |
| `axes` | int 列表 | `(0,)` | 每次 reset 采样 | 标量字段只有 `0` 合法。 |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | 每次 reset 采样 | |
| `operation` | `add` \| `scale` \| `abs` | `abs` | 每次 reset 应用 | |
| `shared_random` | bool | `false` | 每次 reset 采样 | |

### `randomize_encoder_bias`

给 encoder 位置读数加入逐关节常值偏置
（`unilab.envs.mdp.randomize_encoder_bias`）。仅支持 `mode: reset`。写入
的是 entity 数据面（观测偏置），不是物理模型字段，且要求 Torch 控制平面。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `bias_range` | `[lo, hi]` | 必填 | 每次 reset 采样 | 均匀偏置，单位为弧度（关节位置单位）。SAC G1 MuJoCo owner 在全部关节上用 `[-0.01, 0.01]`。 |
| `asset_cfg` | `SceneEntityCfg` | 必填 | 构造期绑定 | 用 `joint_names`/`joint_ids` 选择 joint；至少选中一个关节。 |

## Reset 状态 Term

### `reset_root_state_uniform`

把浮动基 root 重置为 backend 默认状态叠加均匀采样的 SE(3) 偏移
（`unilab.envs.mdp.reset_root_state_uniform`）。固定基或 mocap root 会
fail closed。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `pose_range` | `{x, y, z, roll, pitch, yaw: [lo, hi]}` | 必填 | 每次 reset 采样 | 位置偏移（米）叠加在默认 root 位置与 environment 原点之上；`roll`/`pitch`/`yaw` 偏移（弧度）与默认朝向复合。省略的键默认为 `(0, 0)`。Go2/G1 base owner 用 `x/y: [-0.5, 0.5]` 与 `yaw: [-pi, pi]`。 |
| `velocity_range` | 同上 | 可选（无偏移） | 每次 reset 采样 | 叠加在默认 root 速度上的线速度（m/s）与角速度（rad/s）偏移。Go2/G1 base owner 六个轴都用 `[-0.5, 0.5]`。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | |

### `reset_root_state_uniform_fused`

参数与语义同 `reset_root_state_uniform`
（`unilab.envs.mdp.reset_root_state_uniform_fused`），但在一张编译后的
Torch 图中为所有环境采样，并要求 CUDA tensor runtime 上的 Manager RNG。
backend owner 通过替换现有 `reset_root_state_uniform` term 的 `func` 来
启用，例如 `src/unilab/conf/ppo/task/g1_walk_flat/mjwarp.yaml`。

### `reset_scene_to_default`

`unilab.envs.mdp.reset_scene_to_default`。无参数。把所有已物化的场景实体
重置为 backend 默认 `qpos`/`qvel`；base owner 将其声明为第一个 reset
term。

### `skip_scene_reset`

`unilab.envs.mdp.skip_scene_reset`。无参数。空操作 reset 钩子，作为
MJWarp 的规范 reset 场景 term：backend owner 把继承来的
`reset_scene_to_default` term 的 `func` 换成它（例如
`src/unilab/conf/ppo/task/g1_walk_flat/mjwarp.yaml`、
`src/unilab/conf/ppo/task/go2_joystick_flat/superdex.yaml`）。

## Interval 与 Step Term

### `push_by_setting_velocity`

对 root 施加世界系速度冲击（`unilab.envs.mdp.push_by_setting_velocity`）。
仅支持 `mode: interval`；`interval_range_s` 与 `is_global_time` 声明在
term 上而非 `params` 内。每次应用都会重新读取实时的 `velocity_range`，
curriculum 的范围阶段因此生效。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `velocity_range` | `{x, y, z, roll, pitch, yaw: [lo, hi]}` | 必填 | 每次 interval 触发时采样 | 线速度冲击（`x`/`y`/`z`，m/s）与角速度冲击（`roll`/`pitch`/`yaw`，rad/s）都是世界系增量；省略的键默认为 `(0, 0)`。角速度轴要求 backend 的 interval 角速度能力，在构造期按初始范围绑定。FlashSAC Go2 MuJoCo owner 每 15 s 用 `x/y: [-1.0, 1.0]`、`z: [-0.5, 0.5]`（仅线速度）；SAC G1 owner 每 1-3 s 另加角速度范围（`roll/pitch: [-0.52, 0.52]`、`yaw: [-0.78, 0.78]`）。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | |

### `apply_body_impulse`

按逐环境的"冷却 → 触发 → 持续 → 失效"生命周期施加瞬态随机 body 力 wrench
（`unilab.envs.mdp.apply_body_impulse`）。仅支持 `mode: step`；term 每个
step 运行并自行管理计时器。力矩通道（非零 `torque_range` 或
`body_point_offset`）要求 backend 的 interval body-torque 能力。

| 字段 | 类型 | 默认值/必填 | 生效时机 | 语义与合理取值线索 |
| --- | --- | --- | --- | --- |
| `force_range` | `[lo, hi]` | 必填 | 触发时采样 | 每个力分量的均匀幅值，作用于每个选中 body 的 3 个世界系轴；模型单位（N）。 |
| `torque_range` | `[lo, hi]` | 必填 | 触发时采样 | 世界系力矩（N*m）的同样形状约定。纯力冲击用 `[0.0, 0.0]`。 |
| `duration_s` | `[min, max]` | 必填 | 触发时采样 | wrench 持续时长（秒）；必须满足 `0 <= min <= max`。激活期间每个 step 重新提交。 |
| `cooldown_s` | `[min, max]` | 必填 | 每次触发/reset 后采样 | 下一次冲击前的空闲时长（秒）；`0 <= min <= max`。初始冷却已预采样，因此 `t=0` 不会立即触发。 |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | 构造期绑定 | 用 `body_names`/`body_ids` 选择 body。 |
| `body_point_offset` | `(x, y, z)` | 可选（无） | 触发时应用 | body 连杆坐标系下的施力点；附加 `offset x force` 诱导力矩。要求 packed body-state 读取计划。 |

## 最小配置示例

DR term 声明在所选 backend owner 的 `events:` 段中，文件位于
`src/unilab/conf/<algo>/task/<task>/<backend>.yaml` —— 例如
`src/unilab/conf/flashsac/task/go2_joystick_flat/mujoco.yaml`，以下示例
即改编自该文件的配置段：

```yaml
env:
  events:
    randomize_rigid_body_mass:
      func: unilab.envs.mdp.randomize_rigid_body_mass
      mode: reset
      params:
        asset_cfg:
          _target_: SceneEntityCfg
          name: robot
          body_names: base
        mass_distribution_params: [-1.5, 1.5]
        operation: add
        recompute_inertia: false
    randomize_rigid_body_com:
      func: unilab.envs.mdp.randomize_rigid_body_com
      mode: reset
      params:
        asset_cfg:
          _target_: SceneEntityCfg
          name: robot
          body_names: base
        com_range:
          x: [-0.05, 0.05]
    push_by_setting_velocity:
      func: unilab.envs.mdp.push_by_setting_velocity
      mode: interval
      interval_range_s: [15.0, 15.0]
      is_global_time: true
      params:
        velocity_range:
          x: [-1.0, 1.0]
          y: [-1.0, 1.0]
          z: [-0.5, 0.5]
          roll: [0.0, 0.0]
          pitch: [0.0, 0.0]
          yaw: [0.0, 0.0]
```

用 CLI 选择 backend —— `uv run train --algo <algo> --task <task> --sim
<backend>` —— 并把 backend 未声明支持的 term 从其 owner 中移除（或置为
`null`），因为不支持的 term 会在构造期 fail closed。

## 常见问题

**如何给 Go2 加 friction 随机化？**
在支持该能力的 backend owner 中声明一个 `mode: reset` 的 `geom_friction`
term，用 `geom_names` 正则选中足部 geom。仓库内的模板是 SAC G1 MuJoCo
owner：`src/unilab/conf/sac/task/g1_motion_tracking/mujoco.yaml` 选中
`geom_names: "^(left|right)_foot[1-7]_collision$"`，并用
`ranges: [0.3, 1.2]`、`operation: abs`、`shared_random: true`。请对照 Go2
模型的 geom 名写出等价 pattern。geom friction 是 backend 声明的能力：同一
任务的 Genesis owner 因 Genesis 没有逐环境 friction API 而把
`foot_friction` 置为 `null`，因此不要在未声明该能力的 backend 上启用此
term。

**mass 参数的单位是什么？**
使用模型的 body 质量单位 —— 本仓库的 MuJoCo MJCF 机器人为千克。采样值
经 `operation` 与 backend 声明的默认质量组合：`add` 加千克（FlashSAC Go2
owner 给 `base` 加 ±1.5 kg），`scale` 乘无量纲因子（SAC G1 Genesis owner
把 `pelvis`/`torso_link` 缩放 `[0.9, 1.1]`），`abs` 直接设绝对千克数。
结果按 `min_mass`（默认 `1e-6`）做下限截断。

**Motrix 不支持 gravity 时，如何写一份 Motrix 与 MuJoCo 都能跑的 owner YAML？**
按 {doc}`1-configuration`，gravity reset 随机化目前仅 MuJoCo 可用；
`MotrixBackend` 只声明支持 `base_mass_delta`、`base_com_offset`、`kp`、
`kd` 与 interval push（见 {doc}`0-index`）。共享 term 留在任务的
`base.yaml`，只在 `mujoco.yaml` 声明
`randomize_physics_scene_gravity`；如果共享 base 已声明它，就在
`motrix.yaml` 中设置 `randomize_physics_scene_gravity: null` —— 与
`src/unilab/conf/ppo/task/g1_walk_flat/motrix.yaml` 和 `.../mjwarp.yaml`
中 `pd_gains: null` 的模式相同。若让 Motrix 保留该 term，构造期会 fail
closed 而不是被跳过。

这些 term 背后的 payload 级契约见
{doc}`Domain Randomization Contract </zh_CN/4-developer_guide/2-contracts/4-dr_contract>`。
