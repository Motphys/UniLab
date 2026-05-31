# Scene Composition

Scene composition is a cold-path contract. Env configs describe the scene with
`SceneCfg`; backend materializers convert that declaration into the backend's
native model during initialization.

## Contract

`SceneCfg` and `TerrainSceneCfg` live in `src/unilab/base/scene.py`:

```python
@dataclass
class TerrainSceneCfg:
    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None

@dataclass
class SceneCfg:
    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
```

`EnvCfg.scene` is the single scene source. Static scenes use
`SceneCfg(model_file=...)`. Procedural terrain scenes combine a robot model,
task fragments, and terrain configuration:

```yaml
env:
  scene:
    model_file: src/unilab/assets/robots/go2/go2.xml
    fragment_files:
      - src/unilab/assets/robots/go2/locomotion_task.xml
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
```

The env hands the scene to `create_backend(...)`; it does not call MuJoCo or
Motrix materializers directly.

## Backend Dispatch

`create_backend(...)` in `src/unilab/base/backend/__init__.py` routes on backend
type and applies these rules to the `SceneCfg`:

1. Static `SceneCfg(model_file=...)` with no terrain: load the full static scene
   (merging `fragment_files` on a cold path when present).
2. `SceneCfg(model_file=..., terrain=...)`: treat `model_file` as the robot
   model, assemble the materialized scene on a cold path, and merge
   `fragment_files`.
3. `scene is None`: fail loudly (`create_backend` raises `ValueError`).
4. A backend that does not support a `SceneCfg` feature must raise an explicit
   error. For example, Motrix rough-terrain must fail loudly rather than
   silently fall back to a flat scene.

## MuJoCo Materializer Pipeline

For procedural terrain, the MuJoCo backend calls
`materialize_mujoco_hfield_attached_scene(...)` in
`src/unilab/base/backend/mujoco/xml.py`. The cold-path steps are:

1. `TerrainGenerator(terrain_cfg).write_png(...)` generates a backend-agnostic
   heightfield PNG.
2. `MjSpec.add_hfield(...)` plus `worldbody.add_geom(type=mjGEOM_HFIELD, ...)`
   place the terrain.
3. `MjSpec.from_file(robot_path)` loads the robot spec and `spec.attach(...)`
   attaches it under a frame.
4. Each `fragment_files` entry (task sensors, keyframes) is merged into the
   scene XML.
5. `MjSpec.from_string(...).compile()` returns a compiled `MjModel` together
   with `terrain_origins` (and an optional surface sampler).

The procedural path returns a precompiled `MjModel`; it does not expose a final
`scene.xml` path as a contract. Scene context such as `terrain_origins` is
handed back to the env through backend attributes.

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

User instructions are in {doc}`../../2-user_guide/6-terrain/1-procedural`.
