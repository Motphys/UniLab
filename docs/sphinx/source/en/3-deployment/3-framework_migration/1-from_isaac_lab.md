# Migrating from Isaac Lab

Port an Isaac Lab Manager-Based task to UniLab by keeping its manager and term
structure, then adapting configuration, numeric execution, and scene access at
their owner boundaries. Do not rewrite it as a monolithic `NpEnv` subclass.

This is source-compatible migration, not a promise that an arbitrary Isaac Lab
task runs unchanged. The target path is:

```text
Hydra owner YAML
  -> plain ManagerBasedRlEnvCfg
  -> Registry + make_manager_based_rl_env
  -> ManagerBasedRlEnv on the NumPy/SimBackend runtime
  -> NpEnvState for the existing training and IPC path
```

## Compatibility boundary

```{list-table}
:header-rows: 1
:widths: 28 24 48

* - Isaac Lab surface
  - UniLab status
  - Migration rule
* - Manager categories, term names and dictionary order
  - Compatible
  - Keep observation, action, event, reward, termination, command, and
    curriculum terms in the same order.
* - Function/class terms and `func + params`
  - Compatible
  - Change imports to `unilab.managers`; keep term boundaries and partial
    `reset(env_ids)` semantics.
* - `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`
  - Compatible spelling aliases
  - The canonical UniLab names are `ManagerBasedRlEnv` and
    `ManagerBasedRlEnvCfg`; the aliases point to the same implementation.
* - Tensor values and operations
  - Adapted
  - Replace `torch.Tensor` with `np.ndarray` and use vectorized NumPy. There is
    no manager-facing device API.
* - Nested `@configclass` task configuration
  - Adapted
  - Move the complete task declaration to one Hydra owner YAML. `_target_`
    selects concrete config dataclasses and dotted `func` values select terms.
* - `InteractiveSceneCfg`, USD, and PhysX views
  - Adapted or unsupported
  - Declare a task-owned `SceneCfg` and `EntityCfg`; access state and control
    only through `SceneEntityCfg` and the public entity facade. Unsupported
    capabilities raise during cold-path binding.
* - Omniverse, Isaac renderer, and Torch/PhysX mutation
  - Unsupported
  - UniLab does not install or silently emulate these runtimes.
```

The normative boundary is
{doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`. Only
surfaces backed by registration, configuration, and tests should be described
as compatible.

## Migration procedure

### 1. Inventory the source task

Pin the Isaac Lab revision and list the source manager groups, term names, term
order, parameters, observation dimensions, action dimensions, reset behavior,
and episode timing. Classify each dependency before writing code:

- reuse an existing `unilab.managers` config or `unilab.envs.mdp` term;
- adapt a task-specific term from Torch to NumPy;
- stop if the term requires a capability absent from the public entity or
  `SimBackend` contract.

Do not probe backend objects with `getattr`/`hasattr`, return zeros, or route the
task back to a legacy environment.

### 2. Port scene and assets on the cold path

Replace Isaac Lab's USD/`InteractiveSceneCfg` declaration with a task-owned
`SceneCfg`. Declare every entity and selector needed by terms. The
`SceneEntityCfg` selector resolves names and regular expressions once during
materialization; reset and step reuse cached IDs and NumPy views.

The Cartpole fixture uses a minimal task-owned MJCF asset. More complex assets
must follow
{doc}`scene composition <../../4-developer_guide/1-architecture/4-scene_composition>`
and the selected backend's formal capabilities.

### 3. Port term code, not the manager structure

Keep each function/class term and its parameters. Replace Torch types and
operators mechanically with NumPy, preserve batch shapes, and return one value
per environment where the source term does. Stateful terms resolve selectors
and allocate buffers in their constructor, then update only NumPy buffers on
the hot path.

Python owns term implementations and reusable config dataclasses. It must not
hold a second task-specific list of enabled terms or default weights.

### 4. Make Hydra the only task configuration owner

Declare scene, timing, groups, terms, concrete config types, callables,
parameters, weights, and observation mapping in the owner YAML. For example:

```yaml
env:
  observations:
    policy:
      terms:
        joint_pos_rel:
          func: unilab.envs.mdp.joint_pos_rel
  terminations:
    time_out:
      func: unilab.envs.mdp.time_out
      time_out: true
  policy_observation_group: policy
  critic_observation_group: null

reward:
  alive:
    func: unilab.envs.mdp.is_alive
    weight: 1.0
```

Manager mappings whose value type is a single concrete config dataclass
(observations / events / rewards / terminations / curriculum / metrics /
recorders) may omit `_target_`; materialization infers it from the field type
annotation. `actions` / `commands` have abstract base configs, so they must
still declare a concrete `_target_` (for example
`unilab.envs.mdp.JointPositionActionCfg`). Config classes under
`unilab.managers.` (such as `SceneEntityCfg`) may be referenced by their bare
class name.

Hydra composition materializes this declaration into plain typed config on the
cold path. Unknown fields, unresolved `_target_`/`func` references, and wrong
config types fail before reset or step. Direct Python config construction is
reserved for focused lower-level tests.

### 5. Register one generic runtime path

The task module registers `ManagerBasedRlEnvCfg` and
`make_manager_based_rl_env` for each backend that the repository actually
supports. Backend owner YAMLs carry backend identity and tuning. Users select
the composed owner through the normal CLI, for example:

```bash
uv run train --algo ppo --task <task> --sim mujoco
```

Do not add a task-specific training-script branch, environment factory, runner,
or IPC path.

Two maintainer-approved factory wrappers are the only registered exceptions to
the generic-factory rule: `make_g1_walk_env`
(`src/unilab/tasks/locomotion/g1/manager_terms.py`) constructs the
`G1WalkManagerBasedEnv` subclass that owns the G1 walk manager-based runtime,
and `make_x2_wall_flip_env`
(`src/unilab/tasks/motion_tracking/x2/__init__.py`) resolves untracked X2
meshes on the cold path before delegating to `make_manager_based_rl_env`.
Every other Compatible task registers `make_manager_based_rl_env` directly.

### 6. Validate near each adaptation

Test Hydra composition and typed materialization, term order and math, selector
failure, observation/action shapes, partial reset, and at least one real
registered backend transition. Compare behavior with the pinned source task;
benchmark only after semantic migration is complete.

## Final task status

The #1042 migration closeout covers 39 production tasks and 86 task/backend
registrations. The fail-closed source of truth is
`src/unilab/tasks/migration_matrix.py`: `migration_record()` raises `KeyError`
for a production task name with no entry, so adding a production registration
requires an explicit migration decision.

- 36 tasks are **Compatible** (`target=complete`): the Hydra owner YAML
  materializes the canonical NumPy Manager-Based runtime.
- 3 tasks are **Adapted** (`target=compatibility`): `Go2ArmManipLoco`,
  `SharpaInhandRotation`, and `SharpaInhandRotationGrasp` keep custom
  IK/history or tactile/contact/cache behavior behind one frozen compatibility
  factory each; they migrate only when the formal capability exists.

## Repository evidence

`tests/fixtures/isaac_lab_cartpole/` ports the Manager-Based Cartpole task from
Isaac Lab commit `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`. It preserves all
12 source term names and their order while adapting Torch to NumPy, nested
config objects to Hydra YAML, and the scene/action/reset boundaries to a
fixture-local MJCF implementation. It is test-only evidence, not a production
task or a blanket Isaac Lab support claim.

## See also

- {doc}`Manager-Based API <../../4-developer_guide/1-architecture/6-manager_based_api>`
- {doc}`Environment contract <../../4-developer_guide/2-contracts/1-env_contract>`
- {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`
