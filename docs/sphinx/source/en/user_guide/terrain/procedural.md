# Procedural Terrain


This page only answers four questions:

1. How do I run the rough terrain task that already exists in the current repo?
2. What can and cannot be changed via the Hydra command line?
3. When I want to change the sub-terrain composition, what is the correct entry point?
4. What are the currently known boundaries — not bugs, but constraints?

For the underlying contracts (cold-path materialization, registering a new sub-terrain, hfield export), see the source comments in `src/unilab/base/backend/mujoco/xml.py`, `src/unilab/base/backend/motrix/scene.py`, and `src/unilab/terrains/terrain_generator.py`.

## Current Status

Only one task in the current repo registers and wires up procedural terrain:

| Task | owner YAML | Backend | Entry Algorithm | Code |
| --- | --- | --- | --- | --- |
| `Go2JoystickRough` | `mujoco.yaml`, `motrix.yaml` | MuJoCo / Motrix | PPO (`train_rsl_rl.py`) | `go2/rough.py` |

During env construction:

1. `Go2JoystickRoughCfg` declares a `SceneCfg` whose `model_file` points to `go2.xml`, `fragment_files` brings in the contact sensors from `locomotion_task.xml`, and `scene.terrain` declares an hfield named `terrain_hfield` to be generated.
2. The backend scene materializer calls `TerrainGenerator(...)` to produce a backend-agnostic merged height matrix and `terrain_origins`; the terrain generator itself does not depend on MuJoCo or Motrix.
3. The MuJoCo materializer uses `MjSpec.add_hfield(...)` / `worldbody.add_geom(...)` to create the terrain, then uses `MjSpec.attach(...)` to attach the robot spec to the scene, and finally `compile()` produces the `MjModel`.
4. The Motrix materializer uses `motrixsim.msd.World` to create the terrain world, uses `World.attach(...)` to stitch in the robot world and task fragment, and finally `msd.build(...)` produces the `SceneModel`.
5. `go2.xml` holds the robot-owned `home` keyframe; `locomotion_task.xml` only holds the contact sensors associated with the terrain `floor`.
6. The backend instance owns the cold-path scene artifacts until env `close()`; `terrain_origins` is passed back to env via a backend scene attribute, used for spawn / curriculum.

`step()` / `reset()` / DR provider never read XML or access asset files; everything terrain-related happens on the cold path.

## 1. Direct Training

```bash
# Default single-patch random_rough, critic additionally receives a 17×11 height scan
uv run train --algo ppo --task go2_joystick_rough --sim mujoco
```

The Motrix backend uses the same task owner:

```bash
uv run train --algo ppo --task go2_joystick_rough --sim motrix
```

## 2. Overriding Terrain Parameters via Hydra Command Line

`Go2JoystickRough` explicitly lists a set of override-able fields in `conf/ppo/task/go2_joystick_rough/{mujoco,motrix}.yaml`; these fields allow Hydra struct mode to accept command-line overrides.

| Field | Purpose | YAML Default |
| --- | --- | --- |
| `env.scene.terrain.generator.seed` | Random seed, `null` means re-randomize each time | `42` |
| `env.scene.terrain.generator.curriculum` | `true`: one column per sub-terrain, difficulty increases along rows; `false`: random sampling by `proportion` | `false` |
| `env.scene.terrain.generator.size` | x/y size of a single terrain patch (meters) | `[8.0, 8.0]` |
| `env.scene.terrain.generator.num_rows` | grid row count (in curriculum mode = number of difficulty levels) | `1` |
| `env.scene.terrain.generator.num_cols` | grid column count (ignored in curriculum mode; column count = `len(sub_terrains)`) | `1` |
| `env.scene.terrain.generator.border_width` | width of the flat border around the grid (meters) | `1.0` |
| `env.scene.terrain.generator.difficulty_range` | difficulty sampling range `[min, max]`, ∈ `[0, 1]` | `[0.0, 1.0]` |
| `env.terrain_scan.enabled` | Whether to concatenate the backend-native height scan to the critic obs | `true` |
| `env.terrain_scan.geom_name` | The hfield geom name sampled by the height scan | `floor` |

Example: local small-scale smoke + fixed seed + curriculum mode.

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.scene.terrain.generator.num_rows=4 \
    env.scene.terrain.generator.num_cols=6 \
    env.scene.terrain.generator.seed=42 \
    env.scene.terrain.generator.curriculum=true \
    algo.num_envs=64 algo.max_iterations=2 training.no_play=true
```

Fields not listed in the YAML (e.g. `sub_terrains`) currently **cannot** be overridden from the command line:

- `sub_terrains` is `dict[str, SubTerrainCfg]`, and `SubTerrainCfg` is an abstract base class; rebuilding subclass types from the command line is not safe.
- The default grids of `terrain_scan.measured_points_x` / `terrain_scan.measured_points_y` are defined by the `Go2JoystickRoughCfg` owner; when the scan layout needs to be changed, adjust it explicitly in the owner cfg and validate `obs_groups_spec` against the critic obs shape.

## 3. Modifying Sub-terrains

Sub-terrains are registered in `ALL_TERRAIN_PRESETS` in `unilab.terrains.config`. The 7 sub-terrains mixed by `Go2JoystickRough` by default:

| Name | Implementation | Description |
| --- | --- | --- |
| `flat` | `HfFlatTerrainCfg` | All-zero heightfield, baseline patch |
| `pyramid_stairs` | `HfPyramidStairsTerrainCfg` | Pyramid-shaped ascending stairs (concentric square rings in the heightfield) |
| `pyramid_stairs_inv` | `HfInvertedPyramidStairsTerrainCfg` | Inverted-pyramid descending stairs |
| `hf_pyramid_slope` | `HfPyramidSlopedTerrainCfg` | Heightfield pyramid slope |
| `hf_pyramid_slope_inv` | `HfPyramidSlopedTerrainCfg(inverted=True)` | Inverted pyramid slope |
| `random_rough` | `HfRandomUniformTerrainCfg` | Random uniform noise heightfield |
| `wave_terrain` | `HfWaveTerrainCfg` | Sine wave heightfield |

Each has its own difficulty parameters (`step_height_range`, `slope_range`, `noise_range`, etc.); full field definitions are in `heightfield_terrains.py`. All sub-terrains (including `flat` and stairs) are now implemented via hfield, with resolution uniformly controlled by `TerrainGeneratorCfg.horizontal_scale` / `vertical_scale`.

Built-in compositions are defined in `unilab.terrains.config`, and `Go2JoystickRoughCfg` defines its own owner defaults in `go2/rough.py`:

- `Go2RoughTerrainCfg`: 1 × 1, by default only samples `random_rough` (proportion `0.2`, the rest of the sub-terrains are kept as configurable profiles but default to proportion `0.0`), random mode. Each env instance receives its own independent cfg object.
- `ROUGH_TERRAINS_CFG`: 10 × 20, 7 sub-terrains mixed by proportion, random mode. Currently kept as a reusable profile; not the default training profile of `Go2JoystickRoughCfg`.
- `STAIRS_TERRAINS_CFG`: 10 × 4, curriculum mode, difficulty goes from flat → easy → moderate → challenging. Not referenced by any task at this time; can be used in custom task configs.

## 4. Height Scan Observation

`Go2JoystickRoughEnv` only concatenates the height scan into the `critic` group; the actor obs still follows the 49-dimensional contract of flat Go2 joystick. Default scan points are 17 in the x direction and 11 in the y direction, totaling 187 dimensions, so `obs_groups_spec` is:

| obs group | Dimension | Content |
| --- | ---: | --- |
| `obs` | `49` | actor policy input |
| `critic` | `239` | flat critic 52 dims + height scan 187 dims |

The height scan's geom/body id and sampling offsets are cached during env init; the hot path only calls the backend contract `sample_hfield_height(...)` and consumes cached ids / offsets. XML is not parsed and asset metadata is not read in `step()` / `reset()`.

## 5. Enabling Procedural Terrain in a New Task

A new task enables procedural terrain through `SceneCfg`. `SceneCfg` lives in `src/unilab/base/scene.py`, and `scene.terrain.generator` uses `TerrainGeneratorCfg`.

```yaml
env:
  scene:
    model_file: .../robot.xml
    fragment_files:
      - .../locomotion_task.xml
    terrain:
      kind: hfield
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        seed: 42
        size: [8.0, 8.0]
        num_rows: 10
        num_cols: 20
        border_width: 20.0
```

The env's `__init__` does not need to call the XML materializer directly; hand `scene` over to the backend constructor:

```python
from unilab.base.backend import create_backend

backend = create_backend(..., cfg.scene)
terrain_origins = getattr(backend, "terrain_origins", None)
```

Note: `TerrainGenerator.__init__` mutates the passed cfg in place (writing values into each `sub_cfg.size`). If the same `TerrainGeneratorCfg` instance is shared across multiple envs they will pollute each other; you must use `default_factory` or `copy.deepcopy` to ensure each instance gets its own cfg. `Go2JoystickRoughCfg` handles this via `scene.terrain.generator=Go2RoughTerrainCfg()`.

## 6. Visualization and Offline Replay

To preview the materialized scene without starting training:

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --num_envs 4
```

## 7. Validation

```bash
# Procedural terrain + hfield PNG materializer unit/integration tests
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q

# Hydra compose + Go2JoystickRoughCfg task owner test
uv run pytest tests/config/test_locomotion_params.py -k rough -q

# Go2 rough terrain spawn + height scan contract tests
uv run pytest tests/envs/locomotion/test_go2_terrain_spawn.py tests/envs/locomotion/test_go2_rough_height_scan.py -q

# Hydra command-line override + registry deep-merge loop
uv run pytest tests/config/test_locomotion_params.py \
    -k "apply_cfg_overrides or hydra_terrain_override" -q

# End-to-end smoke: Hydra command-line override of grid size + seed, 2-iter PPO
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.scene.terrain.generator.num_rows=4 env.scene.terrain.generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64

uv run train --algo ppo --task go2_joystick_rough --sim motrix \
    env.scene.terrain.generator.num_rows=4 env.scene.terrain.generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64
```

## Known Constraints

- **Both MuJoCo and Motrix materializers have automated smoke coverage**: the MuJoCo path returns `MjModel`, the Motrix path returns `SceneModel`. Production training performance and convergence quality still need to be recorded by independent benchmarks; they are not guaranteed by smoke tests.
- **The MuJoCo assembly path depends on `MjSpec.attach`**: the robot XML, terrain, and task sensor fragment are assembled at the materialization stage and compiled directly into `MjModel`.
- **The Motrix assembly path depends on `motrixsim.msd.World.attach`**: `go2.xml` holds the keyframe, and `locomotion_task.xml` is wired in as a pure contact-sensor fragment.
- **The height scan is currently only wired into the MuJoCo rough env**: the Motrix rough variant reuses the actor/critic obs of the base `Go2WalkTask`; to wire the height scan into Motrix, the backend `sample_hfield_height(...)` contract must first be aligned.
- **`scene.terrain.generator` is a cold-path config**: modifying the generator after env construction does not affect the already materialized scene. To change terrains, the env must be reconstructed (i.e. rerun the training command).
- **`import unilab.terrains` does not depend on mujoco**: `TerrainGenerator.generate()` / `write_png()` is a pure numpy + imageio path.
