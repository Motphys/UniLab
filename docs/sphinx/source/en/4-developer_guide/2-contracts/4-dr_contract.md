# Domain Randomization Contract

Domain randomization is an env-owner provider contract plus backend capability
application. User configuration examples live in
{doc}`../../2-user_guide/5-domain_randomization/0-index`.

## Lifecycle Classes

- Init lifecycle: changes model identity or geometry. These changes run during
  env/backend initialization, materialization, or cache construction.
- Reset lifecycle: changes state or parameters within the same model identity.
  Providers dispatch a reset randomization payload through `ResetPlan`.
- Interval lifecycle: applies perturbations between steps, such as push or body
  force plans.

Hot paths must not parse XML/assets or probe backend private methods with
`getattr` or `hasattr`.

## Provider Minimum

A task that uses DR should define:

1. A task-owned domain-randomization config dataclass.
2. A `DomainRandomizationProvider`.
3. Reset behavior returning `ResetPlan` state and randomization payloads.
4. Interval behavior through `IntervalRandomizationPlan` when needed.
5. Env construction that calls `self._init_domain_randomization(...)`.

Shared types live in `src/unilab/dr/types.py`, and manager behavior lives in
`src/unilab/dr/manager.py`.

## Backend Capability Boundary

Backend support is explicit. A reset or interval item only counts as a unified
DR item when three pieces exist together:

1. `ResetRandomizationPayload` or `IntervalRandomizationPlan` has an explicit
   field.
2. The backend declares and implements the capability.
3. The task config/provider samples and dispatches that field.

MuJoCo and Motrix differences stay in backend capability declarations,
backend implementations, and owner YAMLs.

## MuJoCo BatchEnvPool Snapshot

Current MuJoCo reset randomization uses `BatchEnvPool.reset(...,
randomization=...)` with a fixed field whitelist. Indexed reads and writes are
available through `get_field_indexed(...)` and `set_field_indexed(...)`. This
interface lives in the `mujoco-uni` package (`mujoco.batch_env`), not in this
repository; the reset-term constants that map onto it are in
`src/unilab/dr/types.py`.

The supported reset fields and their per-env block shapes are below. The leading
dimension is always `len(env_ids)`; the trailing block size is the field's full
flat width in a single `mjModel`.

| Field | Per-env block shape |
| --- | --- |
| `body_mass` | `nbody` |
| `body_ipos` | `3 * nbody` |
| `body_iquat` | `4 * nbody` |
| `body_inertia` | `3 * nbody` |
| `dof_armature` | `nv` |
| `gravity` | `3` |
| `geom_friction` | `3 * ngeom` |
| `kp` | `nu` |
| `kd` | `nu` |

Refresh behavior is fixed by the backend: `body_mass`, `body_ipos`,
`body_iquat`, `body_inertia`, and `dof_armature` trigger an `mj_setConst`
refresh after the write, while `gravity`, `geom_friction`, `kp`, and `kd` do
not.

Two caveats:

- `geom_size` is not in `SUPPORTED_FIELDS`. Geometry size is expressed through
  init-lifecycle model materialization (see `GeomSizeOverride` /
  `ModelVariantSpec` in `src/unilab/dr/types.py`), not reset randomization.
- `gravity` reset randomization requires a `mujoco-uni` build that ships it.
  This repository pins `mujoco-uni==3.8.0`, whose `SUPPORTED_FIELDS` includes
  `gravity`; older packages such as `3.6.0.post6` do not.

## Motor Control Extension

Motor-actuator tasks that do not map policy output directly to backend position
actuators should keep conversion in the env owner layer. Register a pre-step
callback through `SimBackend.set_pre_step_control(...)`; the backend calls it
before physics substeps and refreshes sensors after stepping.

Go2W is the current all-motor actuator example: its env owner combines leg
position targets and wheel torque, while kp/kd randomization stays in the env
owner cache rather than leaking MuJoCo position-actuator mechanics into shared
payloads.

## Evidence In Repo

- DR types: `src/unilab/dr/types.py`
- DR manager: `src/unilab/dr/manager.py`
- Backend interface: `src/unilab/base/backend/base.py`
- Example providers: `src/unilab/envs/locomotion/g1/joystick.py`,
  `src/unilab/envs/motion_tracking/g1/tracking.py`,
  `src/unilab/envs/manipulation/sharpa_inhand/rotation.py`
