# Field Reference

This page is the field-level configuration reference for the domain
randomization (DR) event terms in `src/unilab/envs/mdp/events.py`. It
complements {doc}`0-index`, which audits which registered tasks declare which
terms, and {doc}`1-configuration`, which explains how owner YAMLs are selected.
Every signature, default, and value hint below is taken from the source or
from owner YAMLs under `src/unilab/conf/`; nothing is inferred from design
intent.

DR terms run under three lifecycle classes (see the lifecycle overview in
{doc}`0-index`):

- **construction**: fixed model/tool identity, declared on
  `env.fixed_model_variants`; no event term on this page belongs to it.
- **reset**: event terms with `mode: reset` sample on every episode reset and
  commit through the backend-owned packed reset transaction. `mode: startup`
  is an opt-in reset-lifecycle variant that samples once at initialization and
  replays the frozen per-environment values afterwards.
- **interval**: event terms with `mode: interval` perturb between steps on a
  sampled timer; `mode: step` fires every step and is used by terms that manage
  their own force lifetime.

## Backend Applicability

The exact field-by-backend matrix is owned by the external `unisim-core`
package (`DomainRandomizationCapabilities.supported_reset_terms` /
`supported_interval_terms`) and cannot be confirmed field by field from this
repository, so this page deliberately does not print one. Two rules always
hold:

1. A requested term that the selected backend does not advertise fails closed
   at construction; there is no filtering or silent fallback.
2. Owner YAMLs are per-backend files
   (`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`). A term that one
   backend does not support is set to `null` in that backend's owner while the
   shared base owner keeps it for the others.

In-repo examples of this pattern: `src/unilab/conf/ppo/task/g1_walk_flat/motrix.yaml`
and `.../mjwarp.yaml` set `pd_gains: null` while the MuJoCo owner keeps kp/kd
randomization; `src/unilab/conf/sac/task/g1_motion_tracking/genesis.yaml` sets
`foot_friction: null` with a comment recording that Genesis has no
per-environment friction API. The most complete worked owner example (mass,
COM, gravity, push) is the `events:` block of
`src/unilab/conf/flashsac/task/go2_joystick_flat/mujoco.yaml`.

## Common Parameter Family

These parameters recur across terms. Per-term tables below only list what the
term's constructor actually accepts; unknown params are rejected at
construction.

- `asset_cfg` (`SceneEntityCfg`): selects the entity and its components.
  `name` is required (the scene entity name, e.g. `robot`); components are
  chosen with `body_names`/`body_ids`, `joint_names`/`joint_ids`,
  `geom_names`/`geom_ids`, or `actuator_names`/`actuator_ids`. The `*_names`
  fields accept a regex string; the `*_ids` fields default to "all". Most
  terms default to `SceneEntityCfg("robot")` (the whole robot).
- `distribution`: `"uniform"` (default), `"log_uniform"` (bounds must be
  positive), or `"gaussian"` (the range is read as `(mean, std)` with
  non-negative std). `pd_gains` accepts only `uniform`/`log_uniform`.
- `operation`: how the sample combines with the backend-declared default:
  `"add"` → `default + sample`, `"scale"` → `default * sample`, `"abs"` →
  the sample replaces the default. The default differs per term; see each
  table.
- Range shapes: a scalar pair `[lo, hi]` (e.g. `kp_range`,
  `mass_distribution_params`); an axis dict such as `{x: [lo, hi], ...}`
  (`com_range`, `pose_range`, `velocity_range`; omitted keys default to
  `(0, 0)`); a width-3 pair `[[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]`
  (`gravity_distribution_params`); or, for the model-field terms, `ranges`
  as a flat pair applied to all selected axes, a dict keyed by axis index,
  or a dict of regex → pair matched with `fullmatch` against the selected
  geom/joint names (patterns must each match at least one name and must not
  overlap).

The surrounding term block (`func`, `mode`, `interval_range_s`,
`is_global_time`, `min_step_count_between_reset`, `params`) is an
`EventTermCfg`; `interval_range_s` and `is_global_time` live on the term, not
inside `params`. `interval_range_s` is required for `mode: interval`;
`is_global_time: true` makes all environments share one timer.

## Reset-Lifecycle Terms

### `pd_gains`

Randomizes actuator stiffness `kp` and damping `kd` together
(`unilab.envs.mdp.pd_gains`). Only `mode: reset` is supported; startup and
interval are rejected at construction.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `kp_range` | `[lo, hi]` | required | sampled each reset | Per-actuator stiffness. Finite, `lo <= hi`; must be positive with `log_uniform`. With `operation: scale` the pair is a dimensionless multiplier of the default gains (G1 owners use `[0.9, 1.1]`); with `operation: abs` it is the absolute gain (the Go2 owner uses `kp_range: [31.5, 38.5]`). |
| `kd_range` | `[lo, hi]` | required | sampled each reset | Per-actuator damping, same validation as `kp_range`. Go2 owner: `kd_range: [0.45, 0.55]` with `abs`. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | `actuator_names`/`actuator_ids` select the actuators. |
| `distribution` | `uniform` \| `log_uniform` | `uniform` | sampled each reset | `gaussian` is rejected by this term. |
| `operation` | `scale` \| `abs` | `scale` | applied each reset | `add` is rejected by this term. |

### `randomize_rigid_body_mass`

Randomizes body mass (`unilab.envs.mdp.randomize_rigid_body_mass`). Supports
`mode: reset` and `mode: startup`; startup samples one full-width table at
initialization and replays it.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `asset_cfg` | `SceneEntityCfg` | required | bound at construction | `body_names`/`body_ids` select the bodies. |
| `mass_distribution_params` | `[lo, hi]` | required | sampled each reset (or once at startup) | In the model's body-mass units: kilograms for the MuJoCo MJCF robots in this repo. The FlashSAC Go2 MuJoCo owner uses `[-1.5, 1.5]` with `operation: add` on the `base` body; the SAC G1 Genesis owner uses `[0.9, 1.1]` with `operation: scale` on `pelvis`/`torso_link` at startup. |
| `operation` | `add` \| `scale` \| `abs` | required | applied each reset | No default; must be declared. |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | sampled each reset | `log_uniform` requires positive bounds. |
| `recompute_inertia` | bool | `true` | construction | `true` raises `NotImplementedError` (the reset payload has no inertia-recomputation contract); always set it explicitly to `false`. |
| `min_mass` | float | `1e-6` | applied each reset | Lower clamp for the randomized mass; must be finite and `>= 1e-6`. |

### `randomize_body_mass_inertia`

Scales body mass and principal inertia by one shared per-body factor
(`unilab.envs.mdp.randomize_body_mass_inertia`). Only `mode: reset` is
supported; the factor is sampled once at the first reset, cached per
environment, and reapplied on later resets, so it behaves like a startup term.
The center of mass and the principal frame are left untouched.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `asset_cfg` | `SceneEntityCfg` | required | bound at construction | `body_names`/`body_ids` select the bodies; mass and inertia bindings must cover the same bodies. |
| `scale_range` | `[lo, hi]` | required | sampled once at the first reset | Log-uniform factor `s = e^{2a}` with `a ~ U(ln(lo)/2, ln(hi)/2)`; bounds must be positive. Both mass and principal inertia are multiplied by `s`. |

### `randomize_rigid_body_com`

Additive center-of-mass randomization
(`unilab.envs.mdp.randomize_rigid_body_com`). Supports `mode: reset` and
`mode: startup`. The live `com_range` is re-read on every apply, so a
curriculum can widen the range between resets.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `com_range` | `{x, y, z: [lo, hi]}` | required | sampled each reset (or once at startup) | Offset added to the default body COM position, in meters in the body frame. Omitted axes default to `(0, 0)`. The SAC G1 MuJoCo owner uses `x: [-0.025, 0.025]`, `y/z: [-0.05, 0.05]` on `torso_link`; the FlashSAC Go2 owner uses `x: [-0.05, 0.05]` on `base`. |
| `asset_cfg` | `SceneEntityCfg` | required | bound at construction | `body_names`/`body_ids` select the bodies. |

### `randomize_physics_scene_gravity`

Randomizes the full gravity vector per environment
(`unilab.envs.mdp.randomize_physics_scene_gravity`). Only `mode: reset` is
supported. Prefer a small tilt range: large horizontal gravity can make an
early task unlearnable.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `gravity_distribution_params` | `[[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]` | required | sampled each reset | Per-axis bounds for `(gx, gy, gz)` in m/s^2. With `operation: abs` the sampled vector replaces the default; the FlashSAC Go2 MuJoCo owner declares `[[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]` with `abs` (a zero-width placeholder showing the shape). |
| `operation` | `add` \| `scale` \| `abs` | required | applied each reset | No default; must be declared. |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | sampled each reset | |

### `geom_friction`

Randomizes the 3-component geom friction triple (tangential, torsional,
rolling), matching the pinned mjlab semantics
(`unilab.envs.mdp.geom_friction`). Only `mode: reset` is supported. Final
values must stay non-negative and finite.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `ranges` | pair / axis dict / regex dict | required | sampled each reset | Flat `[lo, hi]` applies to all selected axes; `{0: [lo, hi]}` targets one axis; `{".*_foot.*": [lo, hi]}` applies per-name ranges via regex `fullmatch`. The SAC G1 MuJoCo owner uses `[0.3, 1.2]` with `operation: abs` on the foot collision geoms. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | `geom_names`/`geom_ids` select the geoms. |
| `axes` | list of int | `(0,)` | sampled each reset | Valid axes are `0, 1, 2`; the default touches only the tangential component. |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | sampled each reset | |
| `operation` | `add` \| `scale` \| `abs` | `abs` | applied each reset | |
| `shared_random` | bool | `false` | sampled each reset | `true` gives geoms that share identical bounds one shared sample per environment (used by the SAC G1 MuJoCo owner). |

### `dof_armature`

Randomizes joint armature — the added rotor inertia that models geared
transmissions (`unilab.envs.mdp.dof_armature`). Same parameter family as
`geom_friction`, with a scalar field: only axis `0` is valid. Only
`mode: reset` is supported; final values must stay non-negative and finite.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `ranges` | pair / `{0: [lo, hi]}` / regex dict | required | sampled each reset | Regex keys are matched with `fullmatch` against the selected joint names. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | `joint_names`/`joint_ids` select the joints. |
| `axes` | list of int | `(0,)` | sampled each reset | Only `0` is valid for this scalar field. |
| `distribution` | `uniform` \| `log_uniform` \| `gaussian` | `uniform` | sampled each reset | |
| `operation` | `add` \| `scale` \| `abs` | `abs` | applied each reset | |
| `shared_random` | bool | `false` | sampled each reset | |

### `randomize_encoder_bias`

Adds a constant per-joint bias to the encoder position readings
(`unilab.envs.mdp.randomize_encoder_bias`). Only `mode: reset` is supported.
This writes to the entity data surface (an observation bias), not to a
physics model field, and requires the Torch control plane.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `bias_range` | `[lo, hi]` | required | sampled each reset | Uniform bias in radians (joint position units). The SAC G1 MuJoCo owner uses `[-0.01, 0.01]` on all joints. |
| `asset_cfg` | `SceneEntityCfg` | required | bound at construction | `joint_names`/`joint_ids` select the joints; at least one joint must be selected. |

## Reset-State Terms

### `reset_root_state_uniform`

Resets a floating root to the backend default state plus uniformly sampled
SE(3) offsets (`unilab.envs.mdp.reset_root_state_uniform`). Fixed-base or
mocap roots fail closed.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `pose_range` | `{x, y, z, roll, pitch, yaw: [lo, hi]}` | required | sampled each reset | Position offsets in meters are added on top of the default root position and the environment origin; `roll`/`pitch`/`yaw` offsets in radians compose with the default orientation. Omitted keys default to `(0, 0)`. The Go2/G1 base owners use `x/y: [-0.5, 0.5]` and `yaw: [-pi, pi]`. |
| `velocity_range` | same keys | optional (no offset) | sampled each reset | Linear (m/s) and angular (rad/s) velocity offsets added to the default root velocity. The Go2/G1 base owners use `[-0.5, 0.5]` on all six axes. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | |

### `reset_root_state_uniform_fused`

Same parameters and semantics as `reset_root_state_uniform`
(`unilab.envs.mdp.reset_root_state_uniform_fused`), but samples all
environments in one compiled Torch graph and requires the Manager RNG on a
CUDA tensor runtime. Backend owners swap the `func` of the existing
`reset_root_state_uniform` term, e.g. `src/unilab/conf/ppo/task/g1_walk_flat/mjwarp.yaml`.

### `reset_scene_to_default`

`unilab.envs.mdp.reset_scene_to_default`. No params. Resets all materialized
scene entities to the backend default `qpos`/`qvel`; base owners declare it as
the first reset term.

### `skip_scene_reset`

`unilab.envs.mdp.skip_scene_reset`. No params. No-op reset hook used as the
canonical MJWarp reset scene term: backend owners swap the `func` of the
inherited `reset_scene_to_default` term to it (e.g.
`src/unilab/conf/ppo/task/g1_walk_flat/mjwarp.yaml`,
`src/unilab/conf/ppo/task/go2_joystick_flat/superdex.yaml`).

## Interval And Step Terms

### `push_by_setting_velocity`

Applies a world-frame velocity kick to the root
(`unilab.envs.mdp.push_by_setting_velocity`). Only `mode: interval` is
supported; `interval_range_s` and `is_global_time` are declared on the term,
not in `params`. The live `velocity_range` is re-read on every apply so
curriculum range stages take effect.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `velocity_range` | `{x, y, z, roll, pitch, yaw: [lo, hi]}` | required | sampled at each interval trigger | Linear kicks (`x`/`y`/`z`, m/s) and angular kicks (`roll`/`pitch`/`yaw`, rad/s) are world-frame deltas; omitted keys default to `(0, 0)`. Angular axes require the backend's interval angular-velocity capability, bound at construction from the initial ranges. The FlashSAC Go2 MuJoCo owner uses `x/y: [-1.0, 1.0]`, `z: [-0.5, 0.5]`, linear only, every 15 s; the SAC G1 owners add angular ranges (`roll/pitch: [-0.52, 0.52]`, `yaw: [-0.78, 0.78]`) every 1-3 s. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | |

### `apply_body_impulse`

Applies transient random body wrenches with a per-environment
cooldown → trigger → sustain → expire lifecycle
(`unilab.envs.mdp.apply_body_impulse`). Only `mode: step` is supported; the
term runs every step and manages its own timers. Torque channels (a non-zero
`torque_range` or a `body_point_offset`) require the backend's interval
body-torque capability.

| Field | Type | Default / required | When it applies | Semantics and value hints |
| --- | --- | --- | --- | --- |
| `force_range` | `[lo, hi]` | required | sampled at trigger | Uniform magnitude per force component, applied to the 3 world-frame axes of every selected body; model units (N). |
| `torque_range` | `[lo, hi]` | required | sampled at trigger | Same shape convention for the world-frame torque (N*m). Use `[0.0, 0.0]` for force-only impulses. |
| `duration_s` | `[min, max]` | required | sampled at trigger | How long the wrench is sustained, in seconds; must satisfy `0 <= min <= max`. The wrench is re-staged every step while active. |
| `cooldown_s` | `[min, max]` | required | sampled after each trigger/reset | Idle time before the next impulse, in seconds; `0 <= min <= max`. The initial cooldown is pre-sampled so no impulse fires at `t=0`. |
| `asset_cfg` | `SceneEntityCfg` | `SceneEntityCfg("robot")` | bound at construction | `body_names`/`body_ids` select the bodies. |
| `body_point_offset` | `(x, y, z)` | optional (none) | applied at trigger | Application point in the body link frame; adds the induced `offset x force` torque. Requires the packed body-state read plan. |

## Minimal Configuration Example

DR terms are declared in the `events:` block of the selected backend owner,
`src/unilab/conf/<algo>/task/<task>/<backend>.yaml` — for example
`src/unilab/conf/flashsac/task/go2_joystick_flat/mujoco.yaml`, whose block is
adapted here:

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

Select the backend with the CLI — `uv run train --algo <algo> --task <task>
--sim <backend>` — and keep terms a backend does not advertise out of its
owner (or set them to `null`), because unsupported terms fail closed at
construction.

## FAQ

**How do I add friction randomization for Go2?**
Declare a `geom_friction` term with `mode: reset` in the backend owner that
supports it, selecting the foot geoms with a `geom_names` regex. The SAC G1
MuJoCo owner is the in-repo template:
`src/unilab/conf/sac/task/g1_motion_tracking/mujoco.yaml` selects
`geom_names: "^(left|right)_foot[1-7]_collision$"` and uses
`ranges: [0.3, 1.2]`, `operation: abs`, `shared_random: true`. Check the Go2
model's geom names for the equivalent pattern. Geom friction is a
backend-advertised capability: the Genesis owner for the same task sets
`foot_friction: null` because Genesis has no per-environment friction API, so
do not enable the term on backends that do not advertise it.

**What unit are the mass parameters in?**
The model's body-mass units — kilograms for the MuJoCo MJCF robots in this
repo. The sampled value is combined with the backend-declared default mass by
`operation`: `add` adds kilograms (the FlashSAC Go2 owner adds ±1.5 kg to the
`base` body), `scale` multiplies by a dimensionless factor (the SAC G1
Genesis owner scales `pelvis`/`torso_link` by `[0.9, 1.1]`), and `abs` sets
absolute kilograms. The result is clamped below at `min_mass` (default
`1e-6`).

**How do I keep one task runnable on Motrix and MuJoCo when Motrix has no gravity reset?**
Per {doc}`1-configuration`, gravity reset randomization is MuJoCo-only today;
`MotrixBackend` advertises `base_mass_delta`, `base_com_offset`, `kp`, `kd`,
and interval push only (see {doc}`0-index`). Keep shared terms in the task
`base.yaml`, declare `randomize_physics_scene_gravity` only in `mujoco.yaml`,
and if the shared base declares it, set `randomize_physics_scene_gravity: null`
in `motrix.yaml` — the same pattern as `pd_gains: null` in
`src/unilab/conf/ppo/task/g1_walk_flat/motrix.yaml` and `.../mjwarp.yaml`.
Leaving the term enabled for Motrix would fail closed at construction instead
of being skipped.

For the payload-level contract behind these terms, see
{doc}`Domain Randomization Contract </en/4-developer_guide/2-contracts/4-dr_contract>`.
