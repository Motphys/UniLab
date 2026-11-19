# Termination Terms

Built-in termination terms, compiled from source. All terms return one boolean
per environment. Framework-level terms are re-exported at
`unilab.envs.mdp.<name>`.

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

Terms declared with `time_out: true` in the owner YAML count as episodic
timeouts (truncations) rather than failures; see `TerminationTermCfg.time_out`
in `src/unilab/managers/termination_manager.py`.

## Framework (`unilab.envs.mdp.terminations`)

Source: `src/unilab/envs/mdp/terminations.py`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `time_out`
  - `unilab.envs.mdp.terminations.time_out` (function)
  - —
  - Terminates when the episode length reaches its maximum; declare with
    `time_out: true`.
* - `bad_orientation`
  - `unilab.envs.mdp.terminations.bad_orientation` (function)
  - `limit_angle: float` (required, rad);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Terminates when the root-body tilt angle from upright exceeds
    `limit_angle`.
* - `root_height_below_minimum`
  - `unilab.envs.mdp.terminations.root_height_below_minimum` (function)
  - `minimum_height: float` (required, m);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Terminates when the world-frame root height drops below
    `minimum_height`.
* - `nan_detection`
  - `unilab.envs.mdp.terminations.nan_detection` (function)
  - `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Terminates environments whose root pose/velocity or joint
    position/velocity contains NaN or Inf.
```

## Locomotion — G1 (`unilab.tasks.locomotion.g1.manager_terms`)

Source: `src/unilab/tasks/locomotion/g1/manager_terms.py`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `g1_tilt_exceeded`
  - `unilab.tasks.locomotion.g1.manager_terms.g1_tilt_exceeded` (class)
  - `max_tilt_deg: float` (required, deg)
  - Terminates when the base tilt from upright exceeds `max_tilt_deg`, read
    from the named `torso_upvector` sensor.
* - `g1_base_height_below_minimum`
  - `unilab.tasks.locomotion.g1.manager_terms.g1_base_height_below_minimum`
    (class)
  - `minimum_height: float` (required, m);
    `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")`
  - Terminates when the root body height drops below `minimum_height`, read
    from the packed root-body read plan.
* - `G1WalkTerminationPack`
  - `unilab.tasks.locomotion.g1.manager_terms.G1WalkTerminationPack` (class;
    cfg `G1WalkTerminationPackCfg`)
  - `entity_name: str = "robot"`; `root_body_name: str = "pelvis"`;
    `max_tilt_deg: float = 65.0`; `minimum_height: float = 0.3`
  - Fused canonical G1 walk failure terms evaluated in one carrier read:
    tilt beyond `max_tilt_deg` OR root height below `minimum_height`.
    Declared with `_target_` + `func` (see
    `src/unilab/conf/sac/task/g1_walk_flat/mjwarp.yaml`).
```

## Motion tracking (`unilab.tasks.motion_tracking.common.manager_terms`)

Source: `src/unilab/tasks/motion_tracking/common/manager_terms.py`. All terms
resolve the `MotionCommand` term named by `command_name`.

```{list-table}
:header-rows: 1
:widths: 15 27 33 25

* - Term
  - Function / class
  - Parameters
  - Semantics
* - `BadAnchorPosZOnly`
  - `unilab.tasks.motion_tracking.common.manager_terms.BadAnchorPosZOnly`
    (class)
  - `command_name: str` (required); `threshold: float` (required, m)
  - Terminates when the anchor height gap `|z_motion - z_robot|` exceeds
    `threshold`.
* - `bad_anchor_ori`
  - `unilab.tasks.motion_tracking.common.manager_terms.bad_anchor_ori`
    (function)
  - `command_name: str` (required); `threshold: float` (required);
    `asset_cfg` accepted but unused
  - Terminates when the quaternion-derived z-axis difference between motion
    and robot anchor orientation exceeds `threshold`.
* - `BadMotionBodyPosZOnly`
  - `unilab.tasks.motion_tracking.common.manager_terms.BadMotionBodyPosZOnly`
    (class)
  - `command_name: str` (required); `threshold: float` (required, m);
    `body_names: tuple[str, ...] | None = None`
  - Terminates when any selected body's anchor-relative z error exceeds
    `threshold`.
* - `motion_clip_end`
  - `unilab.tasks.motion_tracking.common.manager_terms.motion_clip_end`
    (function)
  - `command_name: str` (required)
  - Terminates when the motion frame counter reaches the clip end; declare
    with `time_out: true` (see
    `src/unilab/conf/sac/task/g1_motion_tracking/base.yaml`).
```
