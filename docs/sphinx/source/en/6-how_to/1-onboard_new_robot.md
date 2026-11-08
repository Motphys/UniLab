# Walkthrough: Onboard a New Robot

This walkthrough goes from a URDF/MJCF asset to a first PPO training run —
`uv run train --algo ppo --task <task> --sim <backend>` — one verified step at a
time. It composes existing owner layers instead of adding new code paths: assets
({doc}`../2-user_guide/7-tooling/5-robot_import`), a task/scene keyframe,
owner YAMLs ({doc}`../2-user_guide/1-training/2-hydra_config`), and a
registry entry (checklist view: {doc}`../4-developer_guide/3-extending/1-new_task`).

Steps that need a simulator install, a display, or a GPU are marked. The
compose-only checks (`--cfg job`) still pass through the CLI's backend
preflight, so `--sim mujoco` requires the `mujoco` extra; the routed PPO
entrypoint also imports the built-in RL stack, so install the `unirl` extra
as well (`uv sync --extra mujoco --extra unirl` in a source checkout — the
same sync `make setup` runs).

## 0. Pick Names

Three names, used consistently everywhere:

- `<robot>` — asset directory name, e.g. `go2` under
  `src/unilab/assets/robots/go2/`.
- `<task>` — snake_case CLI and owner-YAML directory name, e.g.
  `go2_joystick_flat`. The CLI rejects names with slashes or dots
  (`_check_task_name` in `src/unilab/cli.py`).
- `<RegistryName>` — PascalCase registry key, e.g. `Go2JoystickFlat`.

## 1. Prepare the Assets

Follow {doc}`../2-user_guide/7-tooling/5-robot_import`. For a URDF source:

```bash
uv run scripts/tools/import_robot.py <urdf_path> [robot_name]
```

The script converts URDF through `urdf-to-mjcf`, organizes the meshes, and
opens a `mujoco.viewer` for keyframe tuning (implementation:
`scripts/tools/import_robot.py`). It writes the scene fragment as
`scene.xml`; the fragment name is yours to keep or rename — the Go2
reference uses `scene_flat.xml`. If the source is already MJCF, lay out the
same directory contract by hand:

- `src/unilab/assets/robots/<robot>/assets/` — only `.stl` / `.obj` meshes.
- `src/unilab/assets/robots/<robot>/<robot>.xml` — robot-only MJCF: bodies,
  joints, actuators, sensors, mesh references.
- `src/unilab/assets/robots/<robot>/scene_flat.xml` (the name is yours; the
  import tool writes `scene.xml`) — the task/scene fragment: an `<include>`
  of the robot XML, the ground, task sensors, and the keyframe. Reference
  shape: `src/unilab/assets/robots/go2/scene_flat.xml`.

Prefer primitive collision geometry over visual meshes; high-poly collision
meshes hurt both simulation speed and contact stability.

Verify:

```bash
ls src/unilab/assets/robots/<robot>/
# Expected: assets/  <robot>.xml  and the scene fragment
# (scene.xml from the import tool).
```

## 2. Tune the Keyframe

The default pose is a `<keyframe>` in the task/scene fragment — never in
`<robot>.xml`. This is a hard repository rule: the robot XML stays
task-agnostic so several tasks can share one robot. Reference: the `home`
keyframe of `src/unilab/assets/robots/go2/scene_flat.xml`:

```xml
<keyframe>
  <key name="home" qpos="
      0 0 0.3  1 0 0 0
      0 0.8 -1.5
      ...
      "
    ctrl="..."/>
</keyframe>
```

When the import tool's viewer closes, the tuned keyframe is written into the
scene fragment automatically. When checking it manually, confirm at least:

- Standing pose: the floating-base height puts the feet in normal ground
  contact — not floating, not penetrating. Press `c` in the viewer to show
  contact points.
- Joint positions stay inside their joint ranges and near a natural stance;
  `qpos` is ordered as the free joint (3 position + 4 quaternion) followed by
  the actuated joints in XML order.
- Center of mass: the base sits above the support polygon, so the robot does
  not tip at spawn.

Verify:

```bash
grep -n '<key name=' src/unilab/assets/robots/<robot>/scene_flat.xml  # or scene.xml from the import tool
# Expected: one line per keyframe, e.g. <key name="home" ...>
grep -c '<keyframe' src/unilab/assets/robots/<robot>/<robot>.xml
# Expected: 0 — keyframes stay out of the robot XML.
```

## 3. Write the Owner YAMLs

PPO resolves each task×backend pair to two files (composition rules:
{doc}`../2-user_guide/1-training/2-hydra_config`):

- `src/unilab/conf/ppo/task/<task>/base.yaml` — the shared task declaration.
- `src/unilab/conf/ppo/task/<task>/mujoco.yaml` — the backend owner: inherits
  the base and binds the registry name and the backend identity.

A minimal base owner, adapted from the canonical Go2 owner
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`:

```yaml
# @package _global_
env:
  scene:
    model_file: src/unilab/assets/robots/<robot>/scene_flat.xml  # scene.xml if you kept the import tool's name
    default_keyframe_name: home   # the keyframe from step 2
    entities:
      robot:
        root_body_name: base
        body_names: [base]
        joint_names: [<robot>_hip_joint, ...]   # XML joint names, in action order
        actuator_names: [<robot>_hip, ...]      # XML actuator names
  sim_dt: 0.01
  ctrl_dt: 0.02                   # must be an integer multiple of sim_dt
  max_episode_seconds: 20.0
  observations:
    policy:
      terms:
        base_ang_vel:
          func: unilab.envs.mdp.builtin_sensor
          params: {sensor_name: gyro}
        projected_gravity:
          func: unilab.envs.mdp.projected_gravity_from_sensor
          params: {sensor_name: upvector}
        joint_pos:
          func: unilab.envs.mdp.joint_pos_rel
        joint_vel:
          func: unilab.envs.mdp.joint_vel_rel
        actions:
          func: unilab.envs.mdp.last_action
        command:
          func: unilab.envs.mdp.generated_commands
          params: {command_name: twist}
  actions:
    joint_pos:
      _target_: unilab.envs.mdp.JointPositionActionCfg
      entity_name: robot
      actuator_names: [".*"]
      scale: 0.25
      use_default_offset: true
  commands:
    twist:
      _target_: unilab.envs.mdp.UniformVelocityCommandCfg
      entity_name: robot
      resampling_time_range: [20.0, 20.0]
      heading_command: false
      ranges:
        lin_vel_x: [-0.6, 1.0]
        lin_vel_y: [-0.4, 0.4]
        ang_vel_z: [-0.8, 0.8]
  events:
    reset_scene_to_default:
      func: unilab.envs.mdp.reset_scene_to_default
      mode: reset
    reset_root_state_uniform:
      func: unilab.envs.mdp.reset_root_state_uniform
      mode: reset
      params:
        pose_range:
          x: [-0.5, 0.5]
          y: [-0.5, 0.5]
          yaw: [-3.141592653589793, 3.141592653589793]
        velocity_range:
          x: [-0.5, 0.5]
          y: [-0.5, 0.5]
  terminations:
    time_out:
      func: unilab.envs.mdp.time_out
      time_out: true
    bad_orientation:
      func: unilab.envs.mdp.bad_orientation
      params: {limit_angle: 1.0471975511965976}
  policy_observation_group: policy

reward:
  tracking_lin_vel:
    func: unilab.envs.mdp.track_linear_velocity
    weight: 1.0
    params: {std: 0.5, command_name: twist}
  tracking_ang_vel:
    func: unilab.envs.mdp.track_angular_velocity
    weight: 0.2
    params: {std: 0.5, command_name: twist}
  action_rate:
    func: unilab.envs.mdp.action_rate_l2
    weight: -0.005
```

Notes on the minimal set:

- The sensor names (`gyro`, `upvector`) are the Go2 robot XML's names
  (`src/unilab/assets/robots/go2/go2.xml`). Sensor-binding terms fail closed at
  construction when the name is missing, so declare equivalent sensors in your
  robot XML or drop the terms that bind them.
- The framework tracking terms (`unilab.envs.mdp.track_linear_velocity` /
  `track_angular_velocity`) read root state instead of named sensors, so they
  work on any robot whose command is named `twist`; parameters and semantics
  are in {doc}`../5-reference/6-manager_terms/1-rewards`.
- Add a `critic` observation group plus `env.critic_observation_group: critic`
  when the algorithm should consume privileged observations (the Go2 owner
  does both).

The backend owner (`src/unilab/conf/ppo/task/<task>/mujoco.yaml`), following
`src/unilab/conf/ppo/task/go2_joystick_flat/mujoco.yaml`:

```yaml
# @package _global_
defaults:
  - /task/<task>/base
  - _self_

training:
  task_name: <RegistryName>     # registry key from step 4
  sim_backend: mujoco
algo:
  num_envs: 1024
  max_iterations: 151
```

Verify — config composition only (still passes the CLI's backend preflight):

```bash
uv run train --algo ppo --task <task> --sim mujoco --cfg job \
  | grep -E 'task_name|default_keyframe_name|model_file'
# Expected: the composed values from your two YAMLs.
```

## 4. Register the Task

The registry maps `<RegistryName>` to a config factory plus one env factory per
backend. In-repo tasks register in a package under `src/unilab/tasks/`; the Go2
package is the minimal template
(`src/unilab/tasks/locomotion/go2/__init__.py`):

```python
"""Hydra-owned Manager-Based <robot> production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("<RegistryName>", ManagerBasedRlEnvCfg)
registry.register_env("<RegistryName>", make_manager_based_rl_env, sim_backend="mujoco")
```

Then make `ensure_registries()` import the module by adding it to the
`__unilab_registry_modules__` tuple in `src/unilab/tasks/__init__.py` (or in
your family package's `__init__.py`). Third-party packages outside this repo
register through the `unilab.tasks` entry-point group instead — see
{doc}`../4-developer_guide/1-architecture/5-registry` and `src/unilab/base/registry.py`.

Verify — host-only, no backend needed:

```bash
uv run python -c "
from unilab.base import registry
registry.ensure_registries()
print(registry.list_registered_envs()['<RegistryName>'])
"
# Expected: {'config_factory': 'ManagerBasedRlEnvCfg', 'available_backends': ['mujoco']}
```

## 5. Pick the Backend

Start with `mujoco` (the default backend). Each additional backend needs:

1. a `register_env(..., sim_backend="<backend>")` line in the task package;
2. a matching owner YAML `src/unilab/conf/ppo/task/<task>/<backend>.yaml` with
   `training.sim_backend: <backend>`;
3. evidence that the backend serves the terms your task binds — check the
   generated {doc}`../5-reference/5-support_matrix` and, for DR terms,
   {doc}`../2-user_guide/5-domain_randomization/2-field_reference`.
   Unsupported capabilities fail closed at construction; set a term the
   backend cannot serve to `null` in that backend's owner (pattern:
   `pd_gains: null` in `src/unilab/conf/ppo/task/go2_joystick_flat/motrix.yaml`).

Verify per added backend: repeat the `--cfg job` compose check with
`--sim <backend>`.

## 6. First Training Run

Needs the backend's Python extra plus the `unirl` RL-stack extra (`uv sync
--extra mujoco --extra unirl`); a GPU for any meaningful run. Start small:

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.no_play=true
```

Verified on 1.5.9 (PPO, MuJoCo backend, single RTX 4090) with
`algo.num_envs=16 algo.max_iterations=1` on `go2_joystick_flat` — console
excerpt:

```text
Storing git diff for 'unilab' in: .../logs/rsl_rl_ppo/Go2JoystickFlat/2026-11-08_22-34-23_mujoco/git/unilab.diff
                            Learning iteration 0/1
                            Total steps: 384
                       Steps per second: 1119
                        Collection time: 0.247s
                          Learning time: 0.096s
                        Mean value loss: 0.0143
                    Mean surrogate loss: -0.0449
                      Mean entropy loss: 8.7087
                            Mean reward: 0.02
                    Mean episode length: 1.00
```

First-run checkpoints:

1. Composition and env construction fail fast and name the offending term or
   field — for example, an observation term bound to a missing sensor raises
   at manager construction. Fix the owner YAML and rerun.
2. The run directory appears under `logs/rsl_rl_ppo/<RegistryName>/` — the
   directory level is the `training.task_name` registry name (e.g.
   `Go2JoystickFlat`), not the snake_case CLI task — named
   `<timestamp>_<backend>/` and containing `run_config.json` (details:
   {doc}`../2-user_guide/1-training/3-logging`).
3. Termination counters `Episode_Termination/<term>` show up in TensorBoard
   once episodes complete
   (`uv run tensorboard --logdir logs/rsl_rl_ppo/<RegistryName>`). Per-term
   `reward/<term>` scalars are not forwarded on the PPO route since 1.5.8 —
   the values travel as the fixed-width `info["reward_components"]` tensor
   consumed by the off-policy runtimes instead; see {doc}`2-custom_reward`
   step 3 for how to read per-term values. If any reward or observation goes
   NaN/Inf, training raises with the term name and the NaN guard writes a
   state dump — see {doc}`3-training_diagnostics`.

Then scale back up (the owner values for `algo.num_envs` /
`algo.max_iterations`) and evaluate the latest run:

```bash
uv run eval --algo ppo --task <task> --sim mujoco --load-run -1
```

## Where to Go Next

- Add or tune reward terms: {doc}`2-custom_reward`.
- Built-in term tables: {doc}`../5-reference/6-manager_terms/0-index`.
- Checklist view of the same contract: {doc}`../4-developer_guide/3-extending/1-new_task`.
- Interactive visual smoke check of the spawned scene (needs a display):
  `uv run scripts/visualize_task_env.py --task <RegistryName> --backend mujoco --num_envs 4`.
