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

Shared types live in `unisim.dr.types` (interval term descriptors in
`unisim.dr.interval`); both are re-exported from `src/unilab/dr/__init__.py`.
Manager behavior lives in `src/unilab/dr/manager.py`.

## Backend Capability Boundary

Backend support is explicit. A reset or interval item only counts as a unified
DR item when three pieces exist together:

1. `ResetRandomizationPayload` has an explicit field, or
   `IntervalRandomizationPlan.ops` carries an `IntervalTermOp` for the term.
2. The backend declares and implements the capability.
3. The task config/provider samples and dispatches that field or op.

MuJoCo and Motrix differences stay in backend capability declarations,
backend implementations, and owner YAMLs.

## Interval Term Descriptors

Interval plans are term-descriptor based: `IntervalRandomizationPlan.ops`
carries a tuple of `IntervalTermOp` entries (term name, NumPy payload,
optional `body_ids`) from `unisim.dr.interval`, re-exported through
`unilab.dr`.

- Builtin term names are the `INTERVAL_TERM_*` constants; their payload
  contracts are pinned by `INTERVAL_TERM_SPECS` (`push`: payload shape `(3,)`,
  no `body_ids`; the four body terms: payload shape
  `(num_envs, len(body_ids), 3)` with required `body_ids`).
  `IntervalTermOp.validate()` enforces these contracts for builtin terms;
  unknown custom terms pass validation through untouched.
- Capability ownership stays with the backend:
  `DomainRandomizationCapabilities.supported_interval_terms` is the
  authoritative declaration, queried via `supports_interval_term` /
  `get_unsupported_interval_terms`.
- `DomainRandomizationManager.apply_interval_randomization_if_due` is generic:
  it contains no term names and no per-term branches, so a backend-owned
  custom term needs no manager change. Terms missing from the capability set
  fail closed with `NotImplementedError` naming the backend type and the
  terms; on the backend side, `SimBackend.apply_interval_randomization`
  routes each op through its handler table and fails closed with the backend
  class and term name when no handler exists.
- Ops and plans must stay pickle-safe (protocol 4) across spawn-based
  collector processes: stdlib + NumPy frozen dataclasses only.
- The legacy plan fields (`push_perturbation_limit`, `body_ids`,
  `body_linear_velocity_delta`, `body_angular_velocity_delta`, `body_force`,
  `body_torque`) and the legacy `supports_interval_*` capability bools are
  deprecated: `IntervalRandomizationPlan.iter_ops()` still adapts set legacy
  fields into ops 1:1, and the bools remain as capability fallbacks. New
  providers should populate `ops`; the legacy fields will be removed in the
  next unisim-core major release.

## MuJoCo BatchEnvPool Snapshot

Current MuJoCo reset randomization uses `BatchEnvPool.reset(...,
randomization=...)` with a fixed field whitelist. Indexed reads and writes are
available through `get_field_indexed(...)` and `set_field_indexed(...)`. This
interface lives in the `mujoco-uni-runtime` package (`mujoco_uni.batch_env`), not in this
repository; the reset-term constants that map onto it are in
`unisim.dr.types`.

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
  `ModelVariantSpec` in `unisim.dr.types`), not reset randomization.
- `gravity` reset randomization requires a `mujoco-uni-runtime` build that ships
  it. This repository depends on the official `mujoco` package (`>=3.5`, with
  the default version pinned by `uv.lock`)
  plus `mujoco-uni-runtime`, whose `SUPPORTED_FIELDS` includes `gravity`; older
  batch-env packages such as `mujoco-uni==3.6.0.post6` do not.

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

- DR types: `unisim.dr.types` and `unisim.dr.interval`, re-exported by
  `src/unilab/dr/__init__.py`
- DR manager: `src/unilab/dr/manager.py`
- Backend interface: `unisim.backend.base`
- Example providers: `src/unilab/tasks/locomotion/common/dr_provider.py`,
  `src/unilab/tasks/locomotion/go2_arm/manip_loco.py`,
  `src/unilab/tasks/manipulation/sharpa_inhand/rotation.py`
