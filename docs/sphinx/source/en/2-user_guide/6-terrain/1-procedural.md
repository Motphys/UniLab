# Procedural Terrain

Procedural terrain is a scene-composition capability, not a separate backend
mode. A task owner declares `env.scene.terrain` in Hydra; UniLab validates the
configuration and UniSim's public backend adapters materialize the terrain only
during environment construction.

## Owner declaration

The core reusable pieces are:

- `unilab.base.scene.SceneCfg` for `terrain` and scene files.
- `unilab.terrains` sub-terrain factories for generator composition.

For example, an owner can declare a generated terrain without putting terrain
knowledge in the backend:

```yaml
env:
  scene:
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        _target_: unilab.terrains.TerrainGeneratorCfg
        seed: 42
```

UniLab owns the runtime terrain spawn state in `src/unilab/base/terrain.py`;
UniSim owns generator composition and backend materialization. Task owners that
enable the capability carry their own training evidence.

## Materialization boundary

During `registry.make(...)`:

1. Hydra materializes the selected owner into `ManagerBasedRlEnvCfg`.
2. `create_backend` resolves required robot assets and passes `SceneCfg` to the
   selected UniSim backend.
3. The backend's public scene materializer generates the terrain matrix,
   origins, hfield, and merged scene model.
4. Task owners retain their own scanner bindings and cached IDs.

Step and reset never parse robot XML or inspect asset metadata. They consume
cached IDs and public backend capabilities.

## Owner integration

Select a task owner that declares `env.scene.terrain`, then tune the terrain
grid and seed through Hydra:

```text
env.scene.terrain.generator.num_rows=4
env.scene.terrain.generator.seed=42
```

## Runtime curriculum

With `env.scene.terrain.generator.curriculum=true`, `env.scene.terrain` exposes
`terrain_origins` with shape `(num_rows, num_cols, 3)`, per-environment
`terrain_levels` and `terrain_types`, and the selected `env_origins`. The
origin tensor is updated in place so root reset and cached command references
observe the new tile without rebuilding terrain geometry.

Add the progress term to the owner YAML to unlock rows by training progress:

```yaml
curriculum:
  terrain_levels:
    func: unilab.envs.mdp.terrain_levels_by_progress
    params:
      steps: 10000
```

At each reset, the term samples levels from the unlocked rows only for the
reset environments, preserves their terrain types, and updates their origins
before root-state reset events. `steps` must be a positive integer. Configure a
root reset term such as `unilab.envs.mdp.reset_root_state_uniform` to place the
robot at the selected origin.

## Core validation

```bash
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q
```

Core validation also covers the terrain generator and backend materialization
boundary; ecosystem packages own evidence for additional task families.
