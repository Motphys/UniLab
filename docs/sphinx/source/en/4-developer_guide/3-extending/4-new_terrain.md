# Extending UniLab: New Terrain

Terrain generation is a cold-path scene feature. Keep generation and asset
materialization out of `step()`, `reset()`, and hot domain-randomization loops.

## Implementation Checklist

1. Add a `SubTerrainCfg` implementation in
   `src/unilab/terrains/heightfield_terrains.py` if the terrain needs new
   heightfield geometry.
2. Add a preset in `src/unilab/terrains/config.py` with `@terrain_preset` if the
   terrain should be selectable by name.
3. Wire terrain into a `TerrainGeneratorCfg`, either through an existing named
   set such as `ROUGH_TERRAINS_CFG` or through an owner YAML under `conf/`.
4. Expose terrain to scenes through `SceneCfg.terrain` and `TerrainSceneCfg` in
   `src/unilab/base/scene.py`.
5. If an env needs height observations, create the backend scanner on init via
   `create_hfield_scanner(...)`; read samples through the returned
   `BackendHeightScanner`.
6. Keep terrain spawn, curriculum, and observation dimensions reflected in the
   owning env config and `obs_groups_spec`.

## Validation Near Risk

- Terrain generator shape and numerical behavior:
  `tests/terrains/test_terrain_generator.py`
- Rough locomotion height-scan and spawn behavior:
  `tests/envs/locomotion/test_go2_rough_height_scan.py`,
  `tests/envs/locomotion/test_go2_terrain_spawn.py`,
  `tests/envs/locomotion/test_terrain_spawn.py`
- Backend materialization boundaries: `tests/utils/test_xml_utils.py`

## Evidence In Repo

- Terrain configs and presets: `src/unilab/terrains/config.py`
- Terrain generator: `src/unilab/terrains/terrain_generator.py`
- Heightfield terrain types: `src/unilab/terrains/heightfield_terrains.py`
- Height-scan helper: `src/unilab/envs/locomotion/common/height_scan.py`
