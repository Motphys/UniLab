# Robot Import

This page is the general guide for connecting any new robot asset to UniLab.

Robot import itself is responsible only for assets, model description, naming,
control interfaces, and cold-path materialization; it should not own
task/reward/env semantics.

When adding a new task type, changing observations/rewards, episode logic, or
env owner semantics, see {doc}`../../4-developer_guide/3-extending/1-new_task`.

## Asset Directory Contract

Assets under `src/unilab/assets/robots/<robot_name>/` should follow the
repository format:

- `assets/`: only `.stl` / `.obj` meshes.
- `<robot_name>.xml`: robot-only MJCF containing body / joint / actuator /
  sensor descriptions.
- task/scene fragment: for example `scene.xml`, containing the scene, ground,
  task sensors, and keyframes.

Do not copy the full external package. `<keyframe>` belongs in the task/scene
fragment, not in the robot XML.

## Input Assets

Prefer MuJoCo/MJCF `.xml`, copied according to the contract above.

If the source is URDF-only, convert it with the repository script:

```bash
uv run unilab-import-robot <urdf_path> [robot_name]
```

```{important}
To speed up simulation and improve contact stability, strongly prefer simplified
collision bodies: do not use high-poly visual meshes directly as collision
meshes, and replace collision with box / capsule / sphere / cylinder primitives
where possible.
```

- The default automatic import writes actuators as `position`, which is suitable
  only for position-control owners.
  - If the robot must preserve torque/motor actuator semantics, later task
    extension should follow the control pattern in
    `src/unilab/envs/locomotion/go2w/`: keep action interpretation, PD/torque
    control, and the actuator contract inside the robot owner boundary.
- After conversion, `mujoco.viewer` opens automatically to show the converted
  result and proceed to keyframe adjustment.

## Adjusting The Keyframe

The `home` keyframe is the robot start pose that the integrator must verify
manually.

- In the viewer opened at the end of the previous input-assets step, drag `ctrl`
  in the right-side panel to find a suitable initial pose (`home` keyframe).
- For easier height tuning, the tool expands the freejoint into editable
  xyz/orientation joints in the temporary viewer model and adds a height
  position actuator. Drag the height slider in the right-side control panel to
  adjust base z.
- After the viewer closes, the keyframe is written automatically to `scene.xml`.

When checking `home`, confirm at least:

- The floating-base height puts the feet in normal ground contact, not floating
  or penetrating. Press `c` to show contact points, which helps verify ground
  contact.
- Joint angles stay within joint ranges and are close to a natural standing pose
  or task start pose.

## Output Artifacts

After running `uv run unilab-import-robot <urdf_path> [robot_name]`, the script
generates:

- `src/unilab/assets/robots/<robot_name>/assets/`: converted and organized mesh
  assets.
- `src/unilab/assets/robots/<robot_name>/<robot_name>.xml`: robot MJCF
  description containing only the robot body, joints, actuators, sensors, and
  mesh references.
- `src/unilab/assets/robots/<robot_name>/scene.xml`: task/scene fragment
  containing the `home` keyframe. After the viewer closes, the tuned `home`
  keyframe is written here.

These artifacts only import the robot asset. They do not generate a
task/reward/env owner automatically. Add later task integration through the
corresponding task documentation.
