# Domain Randomization


This page only describes the current domain randomization status of registered tasks in the repo. All conclusions come from the code; nothing is inferred from design intent.

Manager-Based event terms are the only DR declaration path:

- **Manager-Based (Compatible) tasks**: reset / interval randomization is declared through Hydra `events:` manager terms in the owner YAML; reset-lifecycle events sample at reset, interval-lifecycle events perturb between steps. See the `events:` block of `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml` for an example.

The Manager-Based lifecycle is:

- construction path: fixed model/tool identity is attached to `SceneCfg` before backend construction
- reset path: event terms compose writes in `ResetStateTransaction`; the transaction calls `SimBackend.set_state(..., randomization=...)` once
- interval path: event terms submit backend-owned interval plans through the public contract

These three paths correspond to three lifecycle classes:

- **construction-lifecycle identity**: fixed model/tool variants and their immutable env assignment take effect only during backend construction/materialization.
- **reset-lifecycle DR**: items that do not change model identity, only change parameters or reset state within the same model, e.g. `base_mass_delta`, `base_com_offset`, `gravity`, `kp`, `kd`, and backend-declared geometry/model fields.
- **interval-lifecycle DR**: external perturbations between steps, e.g. push.

## Status Conclusions

1. Reset/interval randomization consists of `events:` manager terms in the owner YAML, executed uniformly by the manager lifecycle.
2. Manager-Based owners declare reset behavior through Hydra command/event terms. G1 motion reset perturbations belong to `MotionCommandCfg`, while WBT adds `EventTermCfg` reset and interval terms.
3. `ResetRandomizationPayload` expresses curated reset terms; a backend must advertise every requested term and own its derived-quantity obligation.
4. `MotrixBackend` currently supports `base_mass_delta`, `base_com_offset`, `kp`, `kd`, and interval push; and it requires all model actuators to be position actuators during initialization.
5. Fixed mesh/tool identity is declared by `env.fixed_model_variants`; reset-time geometry fields remain behind backend capability declarations and never change that identity.

## Uniformity Assessment Table

| Task | Declaration path | Structured form? | reset form | interval form | Code |
| --- | --- | --- | --- | --- | --- |
| `Go2JoystickFlat` | Hydra `events:` terms | Yes: owner YAML declares reset events | root-state reset + `pd_gains` kp/kd | none | `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml` |
| `G1WalkFlat` | Hydra `events:` terms | Yes: Hydra `EventTermCfg` + Manager-Based reset terms | root-state reset + kp/kd via `pd_gains` | none | `g1/manager_terms.py` |
| `G1WalkRough` | Hydra `events:` terms | Yes: same Manager-Based event terms as `G1WalkFlat` | root-state reset + kp/kd via `pd_gains` | none | `g1/manager_terms.py` |
| `G1MotionTracking` | Hydra command term | Yes: Hydra `MotionCommandCfg` + Manager-Based command reset | motion frame, root pose/velocity, and joint-position sampling | none | `motion_tracking/common/manager_terms.py` |
| `G1WBTObs` | Hydra `events:` terms | Yes: same motion command + Hydra `EventTermCfg` | motion reset plus mass/COM/PD/friction/encoder-bias events | interval velocity kick | `motion_tracking/g1/manager_terms.py` |
| `AllegroInhandRotation` | Hydra `events:` terms | Yes: Hydra `EventTermCfg` + Manager-Based reset term | entity-scoped hand/ball reset | none | `allegro_inhand/manager_terms.py` |
| `AllegroInhandRotationGrasp` | Hydra `events:` terms | Yes: reuses the rotation reset event + `RecorderTermCfg` | noisy hand reset + grasp collection | none | `allegro_inhand/grasp_gen.py` |

## Per-task Domain Randomization List

| Task | Currently implemented reset domain randomization | Currently implemented interval domain randomization | Default state |
| --- | --- | --- | --- |
| `Go2JoystickFlat` | base xy/yaw and base qvel via `reset_root_state_uniform`; command sampling; kp/kd via `pd_gains` | none | event terms declared and enabled by default in `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml` |
| `G1WalkFlat` | base xy/yaw and base qvel via `reset_root_state_uniform`; command sampling with a planar dead zone; `gait_phase` sampling; kp/kd randomization via `pd_gains` | none | kp/kd enabled on mujoco owners by default; disabled on motrix/mjwarp owners |
| `G1WalkRough` | Same as `G1WalkFlat` (shared owner bases, rough scene) | none | Same defaults as `G1WalkFlat` |
| `G1MotionTracking` | Motion-command frame sampling; root pose perturbation `x/y/z/roll/pitch/yaw`; root velocity perturbation `x/y/z/roll/pitch/yaw`; joint-position noise clipped through the public entity soft limits; action-manager state reset | none | `pose_range`, `velocity_range`, and `joint_position_range` have non-zero perturbations in the base owner |
| `G1WBTObs` | Same motion reset plus base mass, base COM, PD gain, foot friction, and encoder-bias event terms | `push_by_setting_velocity` | The WBT owner explicitly enables all listed event terms; unsupported capabilities raise rather than fall back |
| `AllegroInhandRotation` | Entity-scoped hand/ball reset; an explicitly configured grasp cache is sampled, otherwise `null` explicitly selects the model home pose; optional `joint_noise`, `ball_velocity_noise`, and `ball_z_offset` | none | owner YAML explicitly selects the home pose and zero reset noise; a configured missing or malformed cache fails closed |
| `AllegroInhandRotationGrasp` | Reuses the rotation reset with `joint_noise=0.25`; Manager-Based termination checks fingertip distance, contact count, and ball height; recorder stores successful timeout rows | none | generates the 50k-row Allegro grasp cache and raises `RunComplete` after a successful save |

## Current DR Capabilities and Boundaries

The owner YAML declares event terms; `ResetStateTransaction` composes selected
rows and validates shapes; UniSim backends advertise and apply the curated
payload. Task-specific reset sampling remains owned by command/event terms:

- `G1MotionTracking` pose / velocity / joint noise is owned by its manager command.
- Allegro grasp / object initial-state sampling is task-specific event logic.
- Fixed model/tool identity is construction-time and never reset-time DR.

A requested backend capability that is not advertised fails closed; there is no
filtering or silent fallback.

## Reset gravity Usage

`gravity` is a reset-lifecycle DR: on each reset, a full MuJoCo gravity vector
`(gx, gy, gz)` is sampled per selected environment and dispatched through
`ResetRandomizationPayload.gravity`. Configure it with a reset `EventTermCfg`
that calls `randomize_physics_scene_gravity`; unsupported backends fail closed.
A small tilt range is recommended because large horizontal gravity can make an
early task unlearnable.

## Interval push Usage

Manager-Based tasks configure interval push through the `env.events.push_robot`
term. For example, the retained `g1_wbt_obs` owner uses
`push_by_setting_velocity` with interval and per-axis velocity ranges.

```bash
uv run train --algo sac --task g1_wbt_obs --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

## Fixed Model/Tool Variant Boundary

Manager-Based owners declare fixed variants on the environment config. The task
owns names, source descriptors, and the final assignment; it does not compile a
model:

```yaml
env:
  fixed_model_variants:
    variants:
      - name: tool_a
        source_model_file: tools/tool_a.xml
      - name: tool_b
        source_model_file: tools/tool_b.xml
    # Omit explicit_variant_names for deterministic round-robin assignment.
    explicit_variant_names: [tool_a, tool_b]
```

The Manager factory turns the declaration into a read-only `int32` assignment
with shape `(num_envs,)`; an empty explicit list selects round-robin.
The assignment is task identity: it is fixed after backend construction and is
not resampled by reset events.

UniLab does not open, parse, or compile `source_model_file`, and does not hold
`MjSpec`, `MjModel`, mjbatch, or Warp objects. UniSim adapters own source
realization and must declare `supports_fixed_variants`. Until that
contract lands, a task configured with fixed variants fails closed before env
construction. Heterogeneous variants that cannot expose one public
state/action/sensor layout also fail closed.

Reset-time model-field DR stays inside the selected identity. Its canonical or
per-env baselines come from the backend's declared
`get_reset_term_default(term)` contract; Manager terms do not recompile the
scene or apply a canonical baseline to every tool. When only part of a model
table is randomized, unwritten columns retain the selected environment's
variant baseline.

The ownership boundary and MJWarp/CPU executor split are recorded in
{doc}`ADR-0010 </adr/ADR-0010-fixed-model-variant-ownership-boundary>`.

```{toctree}
:hidden:

1-configuration
```
## Related Tasks

- {doc}`G1 Motion Tracking <../4-tasks/2-motion_tracking>`: confirm motion assets and replay first before enabling DR.
- {doc}`Go2 Rough Terrain <../4-tasks/1-locomotion>`: common items are mass, COM, friction, and push.

For the backend capability boundary, see
{doc}`Domain Randomization Contract </en/4-developer_guide/2-contracts/4-dr_contract>`.
