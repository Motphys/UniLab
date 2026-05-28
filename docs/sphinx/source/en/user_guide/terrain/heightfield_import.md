# Heightfield Import

Heightfield terrain is configured through `SceneCfg` and the terrain generator,
then materialized by the backend on the init path. The committed user-facing
example is `Go2JoystickRough`, with owners in
`conf/ppo/task/go2_joystick_rough/mujoco.yaml` and
`conf/ppo/task/go2_joystick_rough/motrix.yaml`.

## Files To Read

- `src/unilab/terrains/heightfield_terrains.py`
- `src/unilab/terrains/terrain_generator.py`
- `src/unilab/envs/locomotion/go2/rough.py`
- `src/unilab/base/backend/mujoco/xml.py`
- `src/unilab/base/backend/motrix/scene.py`

## Smoke Commands

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_rough/mujoco \
  algo.max_iterations=2 \
  algo.num_envs=64 \
  training.no_play=true

uv run scripts/visualize_task_env.py --task Go2JoystickRough --backend mujoco --num_envs 4
```

Height scan IDs and offsets are cached during env initialization; hot paths call
the backend height-scanner contract instead of parsing XML or asset metadata.
