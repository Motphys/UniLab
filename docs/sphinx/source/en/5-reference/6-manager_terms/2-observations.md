# Observation Terms

Built-in observation terms, compiled from source. Framework-level terms are
re-exported at `unilab.envs.mdp.<name>`.

:::{only} api_ref
```{raw} html
<ul>
  <li>Authoritative signatures: <a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API reference</a>,
      <a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API reference</a>,
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API reference</a></li>
</ul>
```
:::

:::{only} prose_only
Authoritative signatures live in the source tree:
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp),
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks).
:::

## Framework (`unilab.envs.mdp.observations`)

Source: `src/unilab/envs/mdp/observations.py`. Terms that read the physics
state go through the Manager packed Torch read plan; sensor terms bind one
existing named backend sensor at construction and fail closed when the backend
cannot provide it.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `base_lin_vel`
  - `unilab.envs.mdp.observations.base_lin_vel` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Root linear velocity expressed in the root body frame (3).
* - `base_ang_vel`
  - `unilab.envs.mdp.observations.base_ang_vel` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Root angular velocity expressed in the root body frame (3).
* - `projected_gravity`
  - `unilab.envs.mdp.observations.projected_gravity` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - World gravity vector expressed in the root body frame (3).
* - `joint_pos_rel`
  - `unilab.envs.mdp.observations.joint_pos_rel` (function)
  - `biased: bool = False`;
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Joint positions relative to the default pose; `biased=True` additionally
    adds the per-joint encoder bias.
* - `joint_vel_rel`
  - `unilab.envs.mdp.observations.joint_vel_rel` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Joint velocities relative to the default joint velocities.
* - `last_action`
  - `unilab.envs.mdp.observations.last_action` (function)
  - `action_name: str | None = None`
  - Raw actions of the action manager, or of one named action term when
    `action_name` is set.
* - `generated_commands`
  - `unilab.envs.mdp.observations.generated_commands` (function)
  - `command_name: str` (required)
  - Current command vector of the named command term.
* - `builtin_sensor`
  - `unilab.envs.mdp.observations.builtin_sensor` (class)
  - `sensor_name: str` (required); `entity_name: str = "robot"`
  - Reads one existing backend sensor through a cached view.
* - `projected_gravity_from_sensor`
  - `unilab.envs.mdp.observations.projected_gravity_from_sensor` (class)
  - `sensor_name: str` (required; 3-D up-vector sensor);
    `entity_name: str = "robot"`
  - Negated up-vector sensor reading, yielding projected gravity (3).
* - `base_ang_vel_imu_misaligned`
  - `unilab.envs.mdp.observations.base_ang_vel_imu_misaligned` (class)
  - `max_angle_deg: float = 1.0` (deg);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Base angular velocity rotated by a per-env constant IMU mounting
    misalignment, sampled once per env from the Manager RNG and fixed for the
    whole run.
* - `projected_gravity_imu_misaligned`
  - `unilab.envs.mdp.observations.projected_gravity_imu_misaligned` (class)
  - `max_angle_deg: float = 1.0` (deg);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Projected gravity rotated by the same per-env IMU misalignment as
    `base_ang_vel_imu_misaligned` of that env and angle.
```

## Ray scans (`unilab.envs.mdp.ray_observation`)

Source: `src/unilab/envs/mdp/ray_observation.py`. `RayScan` binds a UniSim
tensor ray caster (factory name in `RayPatternCfg.caster`, default
`"uni_ray"`) attached to a scene body or site, traces the declared pattern
every step, and returns per-ray distances; `include_normal` /
`include_hit_point` append 3 values per ray each, flattened. Caster
capabilities are checked at read-phase binding and unsupported features fail
closed. No shipped owner YAML references `RayScan` yet; coverage lives in
`tests/envs/mdp/test_ray_observation.py`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `RayScan`
  - `unilab.envs.mdp.ray_observation.RayScan` (class)
  - `ray_pattern: RayPatternCfg` (required)
  - Per-ray distances from the configured pattern
    (`ray_count` values, plus normals / hit points when enabled).
```

`RayPatternCfg` (frozen dataclass) declares the attachment and the pattern:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Type / default
* - `entity_name`
  - `str` (required); owner entity of the attachment body
* - `pattern`
  - `RayRingPatternCfg | RayGridPatternCfg | RayCameraPatternCfg` (required)
* - `max_distance`
  - `float` (required, > 0)
* - `attachment_body_name` / `attachment_site_name`
  - `str | None`; exactly one of the two must be set
* - `alignment`
  - `"body" | "yaw" | "world"`, default `"body"`
* - `caster`
  - `str`, default `"uni_ray"` (public UniSim caster factory name)
* - `sensor_offset`
  - 3-float sequence, default `(0.0, 0.0, 0.0)`
* - `min_distance`
  - `float = 0.0` (`0 <= min_distance < max_distance`)
* - `include_normal` / `include_hit_point`
  - `bool = False`
* - `sync_geometry`
  - `bool = False`; re-sync backend-owned geometry on reset
* - `allowed_geom_names` / `allowed_tags` / `excluded_body_names` /
    `excluded_ancestor_body_names`
  - `Sequence[str] | None = None`; static ray filters resolved on the cold
    path
* - `caster_options`
  - `dict`, default `{}`; forwarded to the caster factory
```

Pattern configs (all frozen dataclasses; angles in radians unless noted):

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - Pattern
  - Fields
  - Semantics
* - `RayRingPatternCfg`
  - `num_rays: int` (required, > 0); `start: float` (required);
    `stop: float` (required); `pitch: float = 0.0`
  - Rays on a body-local yaw arc with one shared pitch.
* - `RayGridPatternCfg`
  - `size: (float, float)` (required, > 0);
    `resolution: float | (float, float)` (required, > 0);
    `offset: (float, float) = (0.0, 0.0)`;
    `direction: (float, float, float) = (0.0, 0.0, -1.0)` (nonzero)
  - Parallel rays on an inclusive two-dimensional local grid.
* - `RayCameraPatternCfg`
  - `width: int` (required, > 0); `height: int` (required, > 0);
    `fovy: float` (required, degrees in (0, 180));
    `forward: (float, float, float) = (1.0, 0.0, 0.0)`;
    `up: (float, float, float) = (0.0, 0.0, 1.0)` (orthogonal to `forward`)
  - Unit ray directions for a pinhole camera in an explicit local frame.
```

## Locomotion — shared terms

`quadruped_gait_phase` is defined in
`src/unilab/tasks/locomotion/common/manager_terms.py`; the four privileged
per-foot terms are defined in
`src/unilab/tasks/locomotion/common/gait_terms.py` and mirror the mjlab
velocity-task critic observations.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `quadruped_gait_phase`
  - `unilab.tasks.locomotion.common.manager_terms.quadruped_gait_phase`
    (class)
  - `frequency: float = 2.0` (Hz);
    `phase_offsets: list[float] = (0.0, 0.5, 0.5, 0.0)`;
    `command_name: str | None = None`; `command_threshold: float = 0.0`
  - Four-foot phase observation in [0, 1) with the diagonal-trot ordering;
    advances only while the command norm exceeds `command_threshold` when
    `command_name` is set.
* - `foot_height`
  - `unilab.tasks.locomotion.common.gait_terms.foot_height` (class)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")` (foot bodies)
  - World-frame foot body z, shape `(num_envs, num_feet)`; exact on flat
    terrain.
* - `foot_air_time`
  - `unilab.tasks.locomotion.common.gait_terms.foot_air_time` (class)
  - `sensor_groups: list[list[str]]` (required);
    `contact_threshold: float = 0.0`
  - Current per-foot air time in seconds, shape `(num_envs, num_feet)`.
* - `foot_contact`
  - `unilab.tasks.locomotion.common.gait_terms.foot_contact` (class)
  - `sensor_groups: list[list[str]]` (required);
    `contact_threshold: float = 0.0`
  - Per-foot contact flag as float, shape `(num_envs, num_feet)`.
* - `foot_contact_forces`
  - `unilab.tasks.locomotion.common.gait_terms.foot_contact_forces` (class)
  - `sensor_groups: list[list[str]]` (required; 3-D force sensors);
    `contact_threshold: float = 0.0`
  - Log-compressed per-foot contact forces `sign(f) * log1p(|f|)`, shape
    `(num_envs, 3 * num_feet)`.
```

## Locomotion — G1 (`unilab.tasks.locomotion.g1.manager_terms`)

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `G1GaitPhase`
  - `unilab.tasks.locomotion.g1.manager_terms.G1GaitPhase` (class)
  - `frequency: float = 1.5` (Hz); `init_mode: str = "offset_phase"`
    (`"offset_phase"` or `"independent"`)
  - Two-foot gait phase in radians in [0, 2 pi), shape `(num_envs, 2)`;
    phases are resampled per reset and shared with the G1 gait reward terms.
```

## Motion tracking (`unilab.tasks.motion_tracking.common.manager_terms`)

Source: `src/unilab/tasks/motion_tracking/common/manager_terms.py`. All terms
resolve the `MotionCommand` term named by `command_name` and read its
device-resident carriers.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `motion_anchor_pos_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_pos_b`
    (function)
  - `command_name: str` (required)
  - Motion anchor position relative to the robot anchor, expressed in the
    robot anchor frame (3).
* - `motion_anchor_ori_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_anchor_ori_b`
    (function)
  - `command_name: str` (required)
  - Relative anchor orientation (motion w.r.t. robot) as a 6-D rotation
    representation.
* - `robot_body_pos_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.robot_body_pos_b`
    (function)
  - `command_name: str` (required)
  - Robot body positions in the robot anchor frame, flattened to
    `(num_envs, 3 * num_bodies)`.
* - `robot_body_ori_b`
  - `unilab.tasks.motion_tracking.common.manager_terms.robot_body_ori_b`
    (function)
  - `command_name: str` (required)
  - Robot body orientations in the robot anchor frame as 6-D rotations,
    flattened to `(num_envs, 6 * num_bodies)`.
* - `motion_joint_pos_rel`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel`
    (function)
  - `command_name: str` (required)
  - Joint positions relative to the default pose and the per-episode
    `joint_default_bias`.
* - `motion_joint_pos_rel_biased`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel_biased`
    (function)
  - `command_name: str` (required)
  - `motion_joint_pos_rel` plus the per-joint encoder bias.
* - `MotionAnchorPositionObservation`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionAnchorPositionObservation`
    (class)
  - `command_name: str = "motion"`
  - Tensor-native anchor-relative motion position (3), served by the packed
    scene body read.
* - `MotionAnchorOrientationObservation`
  - `unilab.tasks.motion_tracking.common.manager_terms.MotionAnchorOrientationObservation`
    (class)
  - `command_name: str = "motion"`
  - Tensor-native anchor-relative motion orientation as 6-D rotation, served
    by the packed scene body read.
```
