# Scene Export

Dump the resolved scene (joint order, asset paths, friction declarations) for a task owner:

```bash
uv run unilab-export-scene --task g1_motion_tracking --sim motrix \
    --out /tmp/g1_scene.json
```

Useful when porting to real hardware or to debug joint-order mismatches between ONNX export and the motor driver. See {py:mod}`unilab.tools.export_scene`.
