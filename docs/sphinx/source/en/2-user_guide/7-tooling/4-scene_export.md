# Scene Export

Scene export is implemented by `src/unilab/tools/export_scene.py` and registered
as the `unilab-export-scene` console entry in `pyproject.toml`. It accepts a
MuJoCo XML or MJB model path, writes `scene.xml`, copies mesh assets when they
are discoverable, and can create a zip archive.

For task-level materialization checks, use the script that constructs an env
from the registry and owner config:

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --backend mujoco --num_envs 4
```

`tests/test_export_scene.py` covers the export helper, including `scene.xml`
creation, reloadability, and zip output.
