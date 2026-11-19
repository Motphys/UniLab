# Termination Term

内置 termination term，由源码整理。所有 term 每个环境返回一个布尔值。
框架级 term 在 `unilab.envs.mdp.<name>` 处重导出。

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

在 owner YAML 中以 `time_out: true` 声明的 term 计入回合超时（截断）而非
失败；见 `src/unilab/managers/termination_manager.py` 的
`TerminationTermCfg.time_out`。

## 框架级（`unilab.envs.mdp.terminations`）

源码：`src/unilab/envs/mdp/terminations.py`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `time_out`
  - `unilab.envs.mdp.terminations.time_out`（函数）
  - —
  - 回合长度达到上限时终止；声明时加 `time_out: true`。
* - `bad_orientation`
  - `unilab.envs.mdp.terminations.bad_orientation`（函数）
  - `limit_angle: float`（必填，弧度）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 根 body 相对竖直方向的倾角超过 `limit_angle` 时终止。
* - `root_height_below_minimum`
  - `unilab.envs.mdp.terminations.root_height_below_minimum`（函数）
  - `minimum_height: float`（必填，m）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 世界系根高度低于 `minimum_height` 时终止。
* - `nan_detection`
  - `unilab.envs.mdp.terminations.nan_detection`（函数）
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 根位姿/速度或关节位置/速度出现 NaN 或 Inf 的环境终止。
```

## Locomotion —— G1（`unilab.tasks.locomotion.g1.manager_terms`）

源码：`src/unilab/tasks/locomotion/g1/manager_terms.py`。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `g1_tilt_exceeded`
  - `unilab.tasks.locomotion.g1.manager_terms.g1_tilt_exceeded`（类）
  - `max_tilt_deg: float`（必填，角度制）
  - 从具名 `torso_upvector` 传感器读得基座倾角超过 `max_tilt_deg` 时
    终止。
* - `g1_base_height_below_minimum`
  - `unilab.tasks.locomotion.g1.manager_terms.g1_base_height_below_minimum`（类）
  - `minimum_height: float`（必填，m）；
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - 经打包根 body 读计划读得根 body 高度低于 `minimum_height` 时终止。
* - `G1WalkTerminationPack`
  - `unilab.tasks.locomotion.g1.manager_terms.G1WalkTerminationPack`
    （类；配置 `G1WalkTerminationPackCfg`）
  - `entity_name: str = "robot"`；`root_body_name: str = "pelvis"`；
    `max_tilt_deg: float = 65.0`；`minimum_height: float = 0.3`
  - 一次载体读取内融合计算 G1 行走失败条件：倾角超过 `max_tilt_deg`
    或根高度低于 `minimum_height`。以 `_target_` + `func` 声明（见
    `src/unilab/conf/sac/task/g1_walk_flat/mjwarp.yaml`）。
```

## Motion tracking（`unilab.tasks.motion_tracking.common.manager_terms`）

源码：`src/unilab/tasks/motion_tracking/common/manager_terms.py`。所有
term 按 `command_name` 解析 `MotionCommand` term。

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - 函数 / 类
  - 参数
  - 语义
* - `BadAnchorPosZOnly`
  - `unilab.tasks.motion_tracking.common.manager_terms.BadAnchorPosZOnly`（类）
  - `command_name: str`（必填）；`threshold: float`（必填，m）
  - anchor 高度差 `|z_motion - z_robot|` 超过 `threshold` 时终止。
* - `bad_anchor_ori`
  - `unilab.tasks.motion_tracking.common.manager_terms.bad_anchor_ori`（函数）
  - `command_name: str`（必填）；`threshold: float`（必填）；
    `asset_cfg` 可传入但不使用
  - 动作与机器人 anchor 姿态的四元数 z 轴分量差超过 `threshold` 时
    终止。
* - `BadMotionBodyPosZOnly`
  - `unilab.tasks.motion_tracking.common.manager_terms.BadMotionBodyPosZOnly`（类）
  - `command_name: str`（必填）；`threshold: float`（必填，m）；
    `body_names: tuple[str, ...] | None = None`
  - 任一所选 body 的 anchor 相对系 z 误差超过 `threshold` 时终止。
* - `motion_clip_end`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_clip_end`（函数）
  - `command_name: str`（必填）
  - 动作帧计数到达片段末尾时终止；声明时加 `time_out: true`（见
    `src/unilab/conf/sac/task/g1_motion_tracking/base.yaml`）。
```
