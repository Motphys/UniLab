# Extending UniLab: New Task

Start from the contracts: {doc}`../2-contracts/1-env_contract`,
{doc}`../2-contracts/3-task_owner`, and
{doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`.

## Implementation Checklist

1. Pick the closest owner package under `src/unilab/envs/`.
2. Define or extend an env config dataclass and register it with
   `@registry.envcfg("EnvName")`.
3. Register each supported backend implementation with
   `@registry.env("EnvName", sim_backend="mujoco")` or
   `@registry.env("EnvName", sim_backend="motrix")`.
4. If the task lives in a new module, add that module to the package
   `__unilab_registry_modules__` tuple so `ensure_registries()` imports it.
5. Keep `obs_groups_spec` accurate. It must include `obs` and may include
   `critic`; wrappers and learners trust these dimensions.
6. Keep reset and step semantics at the env owner layer:
   `reset(env_indices)` returns `(obs_dict, info_dict)`, and `step(actions)`
   returns `NpEnvState`.
7. Add owner YAMLs under the relevant config root, such as
   `conf/ppo/task/<task>/<backend>.yaml` or
   `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`.
8. Put task or scene keyframes in task/scene XML fragments referenced through
   `SceneCfg.fragment_files`; do not put task-level keyframes in `robot.xml`.

## Validation Near Risk

- Registry and config shape: `tests/base/test_registry.py`,
  `tests/config/test_config_system.py`
- Env observation/reset behavior: `tests/base/test_np_env.py` and the nearest
  task-specific tests under `tests/envs/`
- Script composition: `tests/scripts/test_train_script_configs.py`

## Evidence In Repo

- Registry API: `src/unilab/base/registry.py`
- Env state contract: `src/unilab/base/np_env.py`
- Scene config: `src/unilab/base/scene.py`
- Existing task examples: `src/unilab/envs/locomotion/go2/joystick.py`,
  `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
