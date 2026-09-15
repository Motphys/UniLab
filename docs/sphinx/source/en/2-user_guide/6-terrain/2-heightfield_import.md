# Heightfield Import

Heightfield terrain is configured through `SceneCfg` and the terrain generator,
then materialized by the backend on the init path. The committed user-facing
example is the retained `g1_walk_rough` scene.

## Files To Read

- `src/unilab/terrains/heightfield_terrains.py`
- `unisim.terrain.generator`
- `src/unilab/assets/robots/g1/scene_rough.xml`
- `src/unilab/tasks/locomotion/common/height_scan.py`
- `unisim.backend.mujoco.xml`
- `unisim.backend.motrix.scene`

## Smoke Commands

```bash
uv run train --algo sac --task g1_walk_rough --sim mujoco \
  algo.max_iterations=2 \
  algo.num_envs=64 \
  training.no_play=true
```

Height scan IDs and offsets are cached during env initialization; hot paths call
the backend height-scanner contract instead of parsing XML or asset metadata.
