# Heightfield Import

Heightfield terrain is configured through `SceneCfg` and the terrain generator,
then materialized by the backend on the init path. Core tests exercise the
contract with a generated heightfield scene.

## Files To Read

- `src/unilab/terrains/heightfield_terrains.py`
- `unisim.terrain.generator`
- `src/unilab/tasks/locomotion/common/height_scan.py`
- `tests/utils/test_xml_utils.py`
- `unisim.backend.mujoco.xml`
- `unisim.backend.motrix.scene`

## Smoke Commands

```bash
uv run pytest tests/utils/test_xml_utils.py \
  -k materialize_mujoco_hfield_attached_scene
```

Height scan IDs and offsets are cached during env initialization; hot paths call
the backend height-scanner contract instead of parsing XML or asset metadata.
