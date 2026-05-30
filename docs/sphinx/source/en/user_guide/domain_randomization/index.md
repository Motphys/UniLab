# Domain Randomization


This page only describes the current status of tasks in the repo that are already registered and already wired to a DR provider. All conclusions come from the code; nothing is inferred from design intent.

The current unified entry point lives in `NpEnv._init_domain_randomization()` and `DomainRandomizationManager`:

- init path: the task provider produces an `InitRandomizationPlan`; the manager calls the backend's `apply_init_randomization(...)` during env initialization
- reset path: the task provider produces a `ResetPlan`; the manager validates capability and then calls the backend's `set_state(..., randomization=...)`
- interval path: the task provider produces an `IntervalRandomizationPlan`; the manager calls the backend's `apply_interval_randomization(...)` as needed before step

These three paths correspond to three lifecycle classes:

- **init-lifecycle DR**: items that change the model identity or model geometry; can only take effect during env/backend initialization and materialization, e.g. Sharpa-hand object `geom_size` scaling.
- **reset-lifecycle DR**: items that do not change model identity, only change parameters or reset state within the same model, e.g. `base_mass_delta`, `base_com_offset`, `gravity`, `kp`, `kd`.
- **interval-lifecycle DR**: external perturbations between steps, e.g. push.

## Status Conclusions

1. All tasks currently wired to a DR provider use the unified DR entry point; no task bypasses `DomainRandomizationManager` to run a separate DR flow inside `reset()`.
2. They are all roughly structured: task files define a `domain_rand` config dataclass, a `DomainRandomizationProvider`, and a `ResetPlan`; `G1WalkFlat` reuses `G1Walk`'s provider.
3. What is "unified" today is mainly the entry point and execution flow, not every randomization item itself. The shared helper `build_common_reset_randomization()` currently generates `base_mass_delta`, `base_com_offset`, `gravity`, `kp`, `kd`; the shared interval helper currently only generates push.
4. `ResetRandomizationPayload` can already express `gravity`, `body_iquat`, `body_inertia`, `kp`, `kd`, and `MuJoCoBackend` has declared support. Whether these are actually used still depends on whether the task provider samples and dispatches them.
5. `MotrixBackend` currently supports `base_mass_delta`, `base_com_offset`, `kp`, `kd`, and interval push; and it requires all model actuators to be position actuators during initialization.
6. `geom_size` is not a reset-lifecycle field; Sharpa-hand object geom scale is handled by init-lifecycle model materialization.

## Uniformity Assessment Table

| Task | Uses unified DR entry? | Structured form? | reset form | interval form | Code |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlat` | Yes | Yes: `Domain_Rand + Provider + ResetPlan` | task state sampling + common payload | push | `go1/joystick.py` |
| `Go2JoystickFlat` | Yes | Yes: `Domain_Rand + Provider + ResetPlan` | task state sampling + common payload | push | `go2/joystick.py` |
| `G1WalkFlat` | Yes | Yes: `Domain_Rand + Provider + ResetPlan` | task state sampling + common payload | push | `g1/joystick.py` |
| `G1WalkRough` | Yes | Yes: reuses `G1WalkDomainRandomizationProvider` | task state sampling + common payload | push | `g1/joystick.py` |
| `G1MotionTracking` | Yes | Yes: `Domain_Rand + Provider + ResetPlan` | extensive task-specific reset sampling + common payload | push | `motion_tracking/g1/tracking.py` |
| `AllegroInhandRotation` | Yes | Yes: `DomainRandConfig + Provider + ResetPlan` | task-specific reset sampling + common payload | none | `allegro_inhand/rotation.py` |
| `SharpaInhandRotation` | Yes | Yes: `InitRandomizationPlan + ResetPlan + IntervalRandomizationPlan` | grasp cache sampling + common payload | object `body_force` | `sharpa_inhand/rotation.py` |
| `SharpaInhandRotationGrasp` | Yes | Yes: reuses the Sharpa rotation provider and overrides reset sampling | grasp collection reset + common payload | none | `sharpa_inhand/grasp_gen.py` |

## Per-task Domain Randomization List

| Task | Currently implemented reset domain randomization | Currently implemented interval domain randomization | Default state |
| --- | --- | --- | --- |
| `Go1JoystickFlat` | base xy; base yaw; base qvel; command sampling; `current_actions/last_actions` zeroed; optional `base_mass_delta`; optional `base_com_offset`; optional `gravity` | `push_robots` | `base_mass_delta`, `base_com_offset`, and push enabled by default; `gravity` disabled by default |
| `Go2JoystickFlat` | base xy; base yaw; base qvel; command sampling; `current_actions/last_actions` zeroed; kp/kd randomization (enabled by default); optional `base_mass_delta`; optional `base_com_offset`; optional `gravity` | `push_robots` | kp/kd enabled by default; common payload and push disabled by default |
| `G1WalkFlat` | base xy; base yaw; base qvel sampled by `reset_base_qvel_limit`; command sampling; `gait_phase` sampling; `current_actions/last_actions` zeroed; kp/kd randomization (enabled by default); optional `base_mass_delta`; optional `base_com_offset`; optional `gravity` | `push_robots` | kp/kd enabled by default; common payload and push disabled by default |
| `G1WalkRough` | Same as `G1WalkFlat`, directly reuses the same provider | `push_robots` | kp/kd enabled by default; common payload and push disabled by default |
| `G1MotionTracking` | motion frame sampling; root pose perturbation `x/y/z/roll/pitch/yaw`; root velocity perturbation `x/y/z/roll/pitch/yaw`; joint position noise; under MuJoCo clipped by joint range; `current_actions/last_actions` zeroed; optional `base_mass_delta`; optional `base_com_offset`; optional `gravity` | `push_robots` | `pose_randomization`, `velocity_randomization`, `joint_position_range` have non-zero perturbations by default; common payload and push disabled by default |
| `AllegroInhandRotation` | If a grasp cache exists, sample a grasp randomly; otherwise apply `joint_noise` to hand joints and `ball_z_offset` to the ball; always apply `ball_vel_noise` to ball linear velocity; optional common reset randomization payload (incl. `gravity`) | none | If the grasp cache path is available it is sampled by default; `joint_noise`, `ball_vel_noise`, `ball_z_offset` default to 0; common payload disabled by default |
| `SharpaInhandRotation` | grasp cache bucketed sampling by `scale_ids`; object pose / quat reset; optional common reset randomization payload (incl. `gravity`) | object `body_force` direct force disturbance | `domain_rand.scale_list` defaults come from the owner YAML; under MuJoCo, object geom scale is materialized during init; common payload disabled by default; object force enabled by default via the Sharpa owner YAML |
| `SharpaInhandRotationGrasp` | hand pose reset; object pose / quat reset; collects successful grasps and stores them bucketed by `scale_ids`; optional `base_mass_delta`; optional `base_com_offset`; optional `gravity` | none | Used by default to generate the Sharpa grasp cache; cache filename includes the single scale value; common payload disabled by default |

## Current Unified DR Capabilities and Boundaries

### 1. Unified Entry Point Is Complete

The unified entry point is guaranteed by `NpEnv` and `DomainRandomizationManager`:

- Tasks only need to register a provider
- The manager uniformly performs capability validation
- The backend is uniformly responsible for actually applying the randomization payload

So from an execution-path perspective, the tasks are already unified.

### 2. The Shared Helpers Are Still Narrow

`dr_utils.py` currently has only two classes of shared helpers:

- reset common payload: `base_mass_delta`, `base_com_offset`, `gravity`, `kp`, `kd`
- interval common payload: push

This means:

- Although locomotion tasks all go through the unified entry point, their base xy, yaw, qvel, command, and gait phase are still sampled directly inside each provider
- `G1MotionTracking`'s pose / velocity / joint noise is also task-specific logic
- Allegro's grasp / object initial state sampling is entirely task-specific logic
- Sharpa's `geom_size` scale is init-lifecycle model materialization and is not part of the reset common payload

So today's "uniformity" is more about the contract and the calling convention than "all tasks share the same set of randomization-item schemas".

### 3. Backend Capabilities Already Exceed What Tasks Currently Use

`ResetRandomizationPayload` now contains:

- `base_mass_delta`
- `base_com_offset`
- `gravity`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

Backend capability today:

- `MuJoCoBackend`: supports the 7 reset terms above, plus interval push and interval body force
- `MotrixBackend`: supports `base_mass_delta`, `base_com_offset`, `kp`, `kd`, plus interval push; requires actuators to all be position actuators during initialization

Notes:

- The current `IntervalRandomizationPlan` supports `push_perturbation_limit`, `body_linear_velocity_delta`, and `body_force`; among these, `body_force` expresses hot-path direct external-force perturbations without exposing the backend-private `xfrc_applied` details.
- The current MuJoCo backend's interval push and interval body force are both dispatched through `xfrc_applied`; the Sharpa-hand object disturbance has been switched to direct force disturbance.
- The Motrix backend currently still does not support direct body-force disturbance, so such owner configs must continue to be explicitly disabled.

But on the task side, the current reality is: not every provider constructs these fields. The backend contract is the capability boundary; whether the task config and provider dispatch a payload is what determines whether a given task actually enables the corresponding DR item.

## Reset gravity Usage

`gravity` is a reset-lifecycle DR: on each reset, a full MuJoCo gravity vector `(gx, gy, gz)` is sampled per env subset and dispatched to the backend via `ResetRandomizationPayload.gravity`. This vector expresses both direction and magnitude:

- Direction: determined by the direction of `(gx, gy, gz)`.
- Magnitude: determined by the vector norm `sqrt(gx^2 + gy^2 + gz^2)`.
- Lifecycle: only sampled and written at reset; the env retains that gravity until the next reset re-samples it.
- Backend: currently in UniLab, only the MuJoCo backend declares support for this reset term; the Motrix backend does not. Some tasks filter it by capability and skip it; others raise an error in the validate stage.

The config entry is under each task's `env.domain_rand`:

```yaml
env:
  domain_rand:
    randomize_gravity: true
    gravity_range:
      - [-0.2, -0.2, -10.5]
      - [0.2, 0.2, -8.5]
```

Field semantics:

- `randomize_gravity`: whether to enable gravity reset DR; defaults to `false`.
- `gravity_range`: a `(2, 3)`-shaped per-dimension sampling range; the first and second rows give the upper and lower bounds of each component.
- On each reset, each dimension is uniformly sampled within `[min(row0, row1), max(row0, row1)]`. The direction is not automatically normalized, and the gravity norm is not fixed.

If you only want to randomize the magnitude while keeping the vertical-down direction, only open up the `z` component:

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

If you want to randomize both direction and magnitude, open up `x/y/z`:

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[-0.3,-0.3,-10.5],[0.3,0.3,-8.5]]'
```

Notes:

- `gravity_range` must be convertible into a `(2, 3)` array; otherwise reset will raise an error when constructing the payload.
- This term does not call `mj_setConst`; MuJoCo step / forward reads `mjModel.opt.gravity` directly.
- Do not enable this term under the Motrix backend; the current Motrix capability does not include `gravity`.
- If your current environment still has a `mujoco-uni` package installed that does not include the `gravity` field, MuJoCo reset will raise unsupported field; you need to use a `mujoco-uni` build/release that includes the field.
- During training it is recommended to start from a small tilt range; otherwise sampling a too-large horizontal gravity early on may degrade the task into being unlearnable.

## Interval push Usage

Tasks supporting interval push configure it under `env.domain_rand`:

```yaml
env:
  domain_rand:
    push_robots: true
    push_interval: 750
    max_force: [1.0, 1.0, 0.5]
    push_body_name: null
```

- `push_robots`: whether to enable push.
- `push_interval`: trigger every N env steps.
- `max_force`: a length-3 external-force upper limit; each dimension is sampled within `[-max_force, max_force]`.
- `push_body_name`: the target body / link to apply the force to. Defaults to `null`, meaning the backend's `base_name` is used.

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]' \
  env.domain_rand.push_body_name=torso_link
```

Notes:

- MuJoCo resolves by body name, Motrix resolves by link name; a missing name raises an error during env/backend initialization.
- `push_body_name` is an init config; changing it after env creation does not change the already-resolved target.
- The hot path only samples and applies the external force; it does not parse XML / asset and does not probe backend-private capability.
- MuJoCo push is implemented via `xfrc_applied` external force and does not directly overwrite base velocity.

## `geom_size` Lifecycle Boundary

`geom_size` is explicitly not part of `ResetRandomizationPayload`, and must not be modified on the hot path via `BatchEnvPool.reset(..., randomization=...)`.

The reason is that `geom_size` changes model geometry and model identity; the correct lifecycle is:

1. The task provider generates the model variants and env-to-model assignment in `build_init_randomization_plan(...)`.
2. The MuJoCo backend modifies geom size on the cold path using `MjSpec` and compiles scale-specific `MjModel`s.
3. The backend constructs `BatchEnvPool` with a model sequence of length `num_envs`.

```{toctree}
:hidden:

configuration
writing_providers
```
4. The reset stage only performs state and parameter perturbations within the same model identity; it does not handle `geom_size`.

This boundary exists to honor the cold-path asset/model-metadata access principle: `step()`, `reset()`, and hot-path DR do not parse XML, do not read assets, and do not branch at runtime based on asset metadata.

## Sharpa-hand Object Geom Scale Usage

Sharpa-hand is the current example task for `geom_size` init-lifecycle DR in the repo. Related task configs:

- `task=sharpa_inhand/mujoco`
- `task=sharpa_inhand_grasp/mujoco`

### 1. Config Entry

Sharpa's scale configuration lives in the env owner YAML at `env.domain_rand.scale_list`:

```yaml
env:
  object_body_name: object
  object_geom_name: object
  domain_rand:
    scale_list: [0.5, 0.6, 0.7, 0.8]
```

Field semantics:

- `object_body_name`: the object body name; used to locate the object body during reset / observation, not the target field for scaling.
- `object_geom_name`: the MuJoCo geom name to scale; defaults to `object`.
- `domain_rand.scale_list`: explicit scale list; each value must be greater than 0.
- The order of `domain_rand.scale_list` is the `scale_id` order.
- The length of `domain_rand.scale_list` is the number of model variants.

Each env is statically assigned a `scale_id`. The current assignment rule is contiguous bucket assignment; when `algo.num_envs` is not divisible by `num_scales`, the first few scale buckets get one extra env:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

If `algo.num_envs=4096` and `num_scales=4`, then every 1024 envs use the same scale bucket.

### 2. MuJoCo Materialization Behavior

How the MuJoCo backend applies it:

1. The env/provider builds `ModelVariantSpec` based on `scale_list` during init.
2. The backend uses `MjSpec` to read the model and modify the `size` of the geom corresponding to `object_geom_name`.
3. Each scale compiles a scale-specific `MjModel`.
4. The first time a physics pool is needed, the env-to-model assignment is expanded into a model sequence of length `num_envs`, and then `BatchEnvPool` is constructed.

Therefore, `domain_rand.scale_list` only takes effect during env/backend initialization. Changing `env.domain_rand.scale_list` after env creation does not change the already-materialized model pool.

This flow has three important boundaries:

- `BatchEnvPool` is lazily constructed; the normal path does not first construct a pool for the default model and then rebuild it for `scale_list`.
- Compilation of multiple model variants is done in chunks using process-based parallelism; do not compile in a Python thread, and do not serially compile `num_envs` models in an upper-level for loop.
- Workers compile variants with `MjSpec` and save `.mjb`; the parent process only loads `MjModel.from_binary_path(...)` by `.mjb` path. Do not transmit modified model objects or model bytes back via IPC.

### 3. Grasp Cache and Scale Buckets

The Sharpa rotation task samples from multiple single-scale grasp caches by `scale_ids`:

- The cache filename defaults are jointly determined by `grasp_cache_path` and a single scale value.
- `scale_list: [0.5, 0.6, 0.7, 0.8]` by default corresponds to `cache/sharpa_grasp_linspace_0.5.npy`, `cache/sharpa_grasp_linspace_0.6.npy`, `cache/sharpa_grasp_linspace_0.7.npy`, `cache/sharpa_grasp_linspace_0.8.npy`.
- At rotation startup all cache files for `scale_list` are checked; if any is missing, it errors.
- Each scale bucket only samples from the cache file of its own scale, avoiding mixing grasp initial states across different object scales.

When generating multi-scale caches, run the grasp collection task multiple times separately:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.5]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.6]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.7]' algo.num_envs=4096
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.8]' algo.num_envs=4096
```

Or use the helper in the repo sequentially:

```bash
./scripts/sharpa_collect_grasps.sh 0.5 0.6 0.7 0.8
```

Then train rotation with the same `scale_list`:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco 'env.domain_rand.scale_list=[0.5,0.6,0.7,0.8]' algo.num_envs=4096
```

### 4. Boundaries and Caveats

- `geom_size` is not a reset DR field and must not be written into `ResetPlan.randomization`.
- `BatchEnvPool.reset(..., randomization=...)` currently does not support `geom_size`.
- `geom_size` scale is only materialized under the MuJoCo backend; the Motrix backend currently does not produce a multi-model pool from `scale_list`.
- The length of `scale_list` is the number of model variants, not the number of resamples per reset.
- Each env's `scale_id` is statically assigned during init and does not change at reset.
- When scaling out, scale out the number of model variants; do not compile one model per env according to `num_envs`. Multiple envs share the same `MjModel` corresponding to the same scale bucket.
- The hot path must not read XML, parse assets, or use `getattr` / `hasattr` to probe backend-private capability to decide scaling behavior.
- When extending to other shape DR, prefer to reuse the init-lifecycle contract; do not stuff shape fields into the reset payload.

For configuration examples, see {doc}`configuration`. For the developer
provider interface and backend capability boundary, see
{doc}`writing_providers` and {doc}`Domain Randomization Contract </en/developer_guide/contracts/dr_contract>`.
