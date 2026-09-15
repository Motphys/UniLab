# Procedural Terrain

Procedural terrain is a scene-composition capability, not a separate backend
mode. A task owner declares `env.scene.terrain` in Hydra; UniLab validates the
configuration and UniSim's public backend adapters materialize the terrain only
during environment construction.

## Owner declaration

The core reusable pieces are:

- `unilab.base.scene.SceneCfg` for `terrain` and scene files.
- `unilab.tasks.locomotion.common.rough_manager_terms` for terrain generation,
  rough reset, velocity-command, height-scan, and action owner terms.
- `unilab.tasks.locomotion.common.height_scan` for cached height sampling.

For example, an owner can declare a generated terrain without putting terrain
knowledge in the backend:

```yaml
env:
  scene:
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        _target_: unilab.tasks.locomotion.common.rough_manager_terms.QuadrupedRoughTerrainCfg
        seed: 42
```

UniLab does not retain a procedural-terrain production owner. Generator
composition and backend materialization are covered by core terrain and
materialization tests; task owners that enable the capability carry their own
training evidence.

## Materialization boundary

During `registry.make(...)`:

1. Hydra materializes the selected owner into `ManagerBasedRlEnvCfg`.
2. `create_backend` resolves required robot assets and passes `SceneCfg` to the
   selected UniSim backend.
3. The backend's public scene materializer generates the terrain matrix,
   origins, hfield, and merged scene model.
4. Entity and height-scanner IDs are cached for reset/step use.

Step and reset never parse robot XML or inspect asset metadata. They consume
cached IDs and public backend capabilities.

## Owner integration

Select a task owner that declares `env.scene.terrain`, then tune the terrain
grid and seed through Hydra:

```text
env.scene.terrain.generator.num_rows=4
env.scene.terrain.generator.seed=42
```

## Core validation

```bash
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q
```

Core validation also covers the terrain generator and backend materialization
boundary; ecosystem packages own evidence for additional task families.
