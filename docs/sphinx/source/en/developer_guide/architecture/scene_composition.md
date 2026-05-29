# Scene Composition

Scene composition is a cold-path contract. Env configs describe the scene with
`SceneCfg`; backend materializers convert that declaration into the backend's
native model during initialization.

## Contract

`SceneCfg` lives in `src/unilab/base/scene.py`. Static scenes use
`SceneCfg(model_file=...)`. Procedural terrain scenes combine a robot model,
task fragments, and terrain configuration:

```yaml
env:
  scene:
    model_file: src/unilab/assets/robots/go2/go2.xml
    fragment_files:
      - src/unilab/assets/robots/go2/locomotion_task.xml
    terrain:
      kind: hfield
      hfield_name: terrain_hfield
      geom_name: floor
```

The env hands the scene to `create_backend(...)`; it does not call MuJoCo or
Motrix materializers directly.

## Layer Ownership

| Layer | Owns |
| --- | --- |
| Config / registry | `SceneCfg` fields and owner-YAML override surface |
| Terrain | Backend-agnostic height matrices, terrain origins, and terrain presets |
| Backend materializer | XML/world assembly, native model compilation, scene artifact cleanup |
| Env | MDP semantics, reset, reward, observation, and cached scene context use |

## Cold-Path Boundary

Allowed on cold paths:

- Reading XML and asset files.
- Generating terrain heightfields.
- Compiling MuJoCo `MjModel` or Motrix scene models.
- Resolving scene IDs, terrain origins, and scanner handles.

Disallowed on hot paths:

- Parsing XML or reading assets during `step`, `reset`, or interval DR.
- Branching reward or observation logic on raw asset metadata.
- Probing backend-private scene methods instead of using explicit contracts.
- Regenerating terrain after env construction.

## Go2 Rough Terrain Evidence

The current procedural terrain user-facing path is Go2 rough terrain:

- Env owner: `src/unilab/envs/locomotion/go2/rough.py`
- Terrain generator: `src/unilab/terrains/terrain_generator.py`
- MuJoCo materializer: `src/unilab/base/backend/mujoco/xml.py`
- Motrix materializer: `src/unilab/base/backend/motrix/scene.py`
- Owner YAMLs: `conf/ppo/task/go2_joystick_rough/mujoco.yaml`,
  `conf/ppo/task/go2_joystick_rough/motrix.yaml`

User instructions are in {doc}`../../user_guide/terrain/procedural`.
