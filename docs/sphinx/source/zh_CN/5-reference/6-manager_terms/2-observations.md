# Observation Term

内置 observation term，由源码整理。框架级 term 在
`unilab.envs.mdp.<name>` 处重导出。

:::{only} api_ref
```{raw} html
<ul>
  <li>权威签名见 <a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API 参考</a>、
      <a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API 参考</a>、
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API 参考</a>。</li>
</ul>
```
:::

:::{only} prose_only
权威签名见源码树：
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp)、
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks)。
:::

## 框架级（`unilab.envs.mdp.observations`）

源码：`src/unilab/envs/mdp/observations.py`。读取物理状态的 term 走
Manager 打包的 Torch read plan；传感器 term 在构造期绑定一个已存在的具名
后端传感器，后端不具备该能力时直接抛错（fail-closed）。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `base_lin_vel`
  - `unilab.envs.mdp.observations.base_lin_vel`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 根线速度在根 body 系下的表示（3 维）。
* - `base_ang_vel`
  - `unilab.envs.mdp.observations.base_ang_vel`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 根角速度在根 body 系下的表示（3 维）。
* - `projected_gravity`
  - `unilab.envs.mdp.observations.projected_gravity`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 世界重力向量在根 body 系下的表示（3 维）。
* - `joint_pos_rel`
  - `unilab.envs.mdp.observations.joint_pos_rel`（函数）
  - `biased: bool = False`；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 关节位置相对默认位姿；`biased=True` 时额外加上逐关节编码器偏置。
* - `joint_vel_rel`
  - `unilab.envs.mdp.observations.joint_vel_rel`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 关节速度相对默认关节速度。
* - `last_action`
  - `unilab.envs.mdp.observations.last_action`（函数）
  - `action_name: str | None = None`
  - 动作管理器的原始动作；设置 `action_name` 时只取该具名动作 term。
* - `generated_commands`
  - `unilab.envs.mdp.observations.generated_commands`（函数）
  - `command_name: str`（必填）
  - 具名指令 term 的当前指令向量。
* - `builtin_sensor`
  - `unilab.envs.mdp.observations.builtin_sensor`（类）
  - `sensor_name: str`（必填）；`entity_name: str = "robot"`
  - 通过缓存视图读取一个已存在的后端传感器。
* - `projected_gravity_from_sensor`
  - `unilab.envs.mdp.observations.projected_gravity_from_sensor`（类）
  - `sensor_name: str`（必填；3 维朝上向量传感器）；
    `entity_name: str = "robot"`
  - 朝上向量传感器读数取负，得到投影重力（3 维）。
* - `base_ang_vel_imu_misaligned`
  - `unilab.envs.mdp.observations.base_ang_vel_imu_misaligned`（类）
  - `max_angle_deg: float = 1.0`（度）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 基座角速度叠加逐 env 恒定的 IMU 安装偏差旋转；偏差由 Manager RNG
    采样一次并在整个运行期间固定。
* - `projected_gravity_imu_misaligned`
  - `unilab.envs.mdp.observations.projected_gravity_imu_misaligned`（类）
  - `max_angle_deg: float = 1.0`（度）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 投影重力叠加与同一 env、同一角度下
    `base_ang_vel_imu_misaligned` 相同的 IMU 安装偏差。
```

## 射线扫描（`unilab.envs.mdp.ray_observation`）

源码：`src/unilab/envs/mdp/ray_observation.py`。`RayScan` 绑定一个
UniSim 张量射线投射器（工厂名见 `RayPatternCfg.caster`，默认
`"uni_ray"`），挂接在场景 body 或 site 上，每步按声明的图样投射并返回逐
射线距离；`include_normal` / `include_hit_point` 各自为每条射线追加 3 个
值并展平。投射器能力在读阶段绑定时检查，不支持的特性 fail-closed。当前
没有随仓库发布的 owner YAML 引用 `RayScan`；覆盖见
`tests/envs/mdp/test_ray_observation.py`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `RayScan`
  - `unilab.envs.mdp.ray_observation.RayScan`（类）
  - `ray_pattern: RayPatternCfg`（必填）
  - 按配置图样返回逐射线距离（`ray_count` 个值；启用时追加法向 / 命中
    点）。
```

`RayPatternCfg`（冻结 dataclass）声明挂接方式与图样：

```{list-table}
:header-rows: 1
:widths: 30 70

* - 字段
  - 类型 / 默认值
* - `entity_name`
  - `str`（必填）；挂接 body 所属的 entity
* - `pattern`
  - `RayRingPatternCfg | RayGridPatternCfg | RayCameraPatternCfg`（必填）
* - `max_distance`
  - `float`（必填，> 0）
* - `attachment_body_name` / `attachment_site_name`
  - `str | None`；两者必须恰好设置其一
* - `alignment`
  - `"body" | "yaw" | "world"`，默认 `"body"`
* - `caster`
  - `str`，默认 `"uni_ray"`（UniSim 公开投射器工厂名）
* - `sensor_offset`
  - 3 元浮点序列，默认 `(0.0, 0.0, 0.0)`
* - `min_distance`
  - `float = 0.0`（满足 `0 <= min_distance < max_distance`）
* - `include_normal` / `include_hit_point`
  - `bool = False`
* - `sync_geometry`
  - `bool = False`；reset 时重新同步后端持有的几何
* - `allowed_geom_names` / `allowed_tags` / `excluded_body_names` /
    `excluded_ancestor_body_names`
  - `Sequence[str] | None = None`；冷路径解析的静态射线过滤器
* - `caster_options`
  - `dict`，默认 `{}`；透传给投射器工厂
```

图样配置（均为冻结 dataclass；除注明外角度均为弧度）：

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - 图样
  - 字段
  - 语义
* - `RayRingPatternCfg`
  - `num_rays: int`（必填，> 0）；`start: float`（必填）；
    `stop: float`（必填）；`pitch: float = 0.0`
  - body 局部系下一段 yaw 圆弧上的射线，共享同一俯仰角。
* - `RayGridPatternCfg`
  - `size: (float, float)`（必填，> 0）；
    `resolution: float | (float, float)`（必填，> 0）；
    `offset: (float, float) = (0.0, 0.0)`；
    `direction: (float, float, float) = (0.0, 0.0, -1.0)`（非零）
  - 局部二维含端点网格上的平行射线。
* - `RayCameraPatternCfg`
  - `width: int`（必填，> 0）；`height: int`（必填，> 0）；
    `fovy: float`（必填，角度制，范围 (0, 180)）；
    `forward: (float, float, float) = (1.0, 0.0, 0.0)`；
    `up: (float, float, float) = (0.0, 0.0, 1.0)`（须与 `forward`
    正交）
  - 显式局部系下针孔相机的单位射线方向。
```

## Locomotion —— 通用 term

`quadruped_gait_phase` 定义于
`src/unilab/tasks/locomotion/common/manager_terms.py`；四个特权（critic）
逐脚观测 term 定义于 `src/unilab/tasks/locomotion/common/gait_terms.py`，
对应 mjlab velocity 任务 critic 的观测。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `quadruped_gait_phase`
  - `unilab.tasks.locomotion.common.manager_terms.quadruped_gait_phase`
    （类）
  - `frequency: float = 2.0`（Hz）；
    `phase_offsets: list[float] = (0.0, 0.5, 0.5, 0.0)`；
    `command_name: str | None = None`；`command_threshold: float = 0.0`
  - 四脚相位观测，取值 [0, 1)，对角小跑步态顺序；设置 `command_name`
    后仅当指令模长超过 `command_threshold` 时推进。
* - `foot_height`
  - `unilab.tasks.locomotion.common.gait_terms.foot_height`（类）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`（脚部 body）
  - 世界系脚部 body 高度 z，形状 `(num_envs, num_feet)`；平地上精确。
* - `foot_air_time`
  - `unilab.tasks.locomotion.common.gait_terms.foot_air_time`（类）
  - `sensor_groups: list[list[str]]`（必填）；
    `contact_threshold: float = 0.0`
  - 当前逐脚腾空时间（秒），形状 `(num_envs, num_feet)`。
* - `foot_contact`
  - `unilab.tasks.locomotion.common.gait_terms.foot_contact`（类）
  - `sensor_groups: list[list[str]]`（必填）；
    `contact_threshold: float = 0.0`
  - 逐脚接触标志（浮点），形状 `(num_envs, num_feet)`。
* - `foot_contact_forces`
  - `unilab.tasks.locomotion.common.gait_terms.foot_contact_forces`（类）
  - `sensor_groups: list[list[str]]`（必填；3 维力传感器）；
    `contact_threshold: float = 0.0`
  - 对数压缩的逐脚接触力 `sign(f) * log1p(|f|)`，形状
    `(num_envs, 3 * num_feet)`。
```

## Locomotion —— G1（`unilab.tasks.locomotion.g1.manager_terms`）

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `G1GaitPhase`
  - `unilab.tasks.locomotion.g1.manager_terms.G1GaitPhase`（类）
  - `frequency: float = 1.5`（Hz）；
    `init_mode: str = "offset_phase"`（`"offset_phase"` 或
    `"independent"`）
  - 双脚步态相位（弧度，[0, 2 pi)），形状 `(num_envs, 2)`；每次 reset
    重采样，并与 G1 步态奖励 term 共享。
```

## Motion tracking（`unilab.tasks.motion_tracking.common.manager_terms`）

源码：`src/unilab/tasks/motion_tracking/common/manager_terms.py`。所有
term 按 `command_name` 解析 `MotionCommand` term，并读取其驻留设备的
carrier。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `motion_anchor_pos_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_pos_b`（函数）
  - `command_name: str`（必填）
  - 动作 anchor 相对机器人 anchor 的位置，在机器人 anchor 系下表示
    （3 维）。
* - `motion_anchor_ori_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_ori_b`（函数）
  - `command_name: str`（必填）
  - anchor 相对姿态（动作相对机器人）的 6 维旋转表示。
* - `robot_body_pos_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.robot_body_pos_b`（函数）
  - `command_name: str`（必填）
  - 机器人各 body 位置在机器人 anchor 系下表示，展平为
    `(num_envs, 3 * num_bodies)`。
* - `robot_body_ori_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.robot_body_ori_b`（函数）
  - `command_name: str`（必填）
  - 机器人各 body 姿态在机器人 anchor 系下的 6 维旋转表示，展平为
    `(num_envs, 6 * num_bodies)`。
* - `motion_joint_pos_rel`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel`（函数）
  - `command_name: str`（必填）
  - 关节位置相对默认位姿与逐回合 `joint_default_bias`。
* - `motion_joint_pos_rel_biased`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel_biased`（函数）
  - `command_name: str`（必填）
  - 在 `motion_joint_pos_rel` 基础上加入逐关节编码器偏置。
* - `MotionAnchorPositionObservation`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionAnchorPositionObservation`（类）
  - `command_name: str = "motion"`
  - 张量原生 anchor 相对动作位置（3 维），由打包场景 body 读取提供。
* - `MotionAnchorOrientationObservation`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionAnchorOrientationObservation`（类）
  - `command_name: str = "motion"`
  - 张量原生 anchor 相对动作姿态（6 维旋转表示），由打包场景 body
    读取提供。
```
