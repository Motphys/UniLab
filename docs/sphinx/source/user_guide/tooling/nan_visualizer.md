# NaN Visualizer

If a training run dies with `NaN` in reward or observation, use:

```bash
uv run unilab-viz-nan --run-dir runs/<run>/
```

This walks the last N steps and highlights which env / which observation key first produced a NaN. See {py:mod}`unilab.tools.viz_nan`.
