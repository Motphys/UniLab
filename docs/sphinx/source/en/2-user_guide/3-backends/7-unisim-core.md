# UniSim Core migration

UniLab physics backends are being moved to the independent `unisim-core`
package. The distribution and import names are deliberately different:

```bash
uv sync --extra mujoco
uv run python -c "import unisim; print(unisim.ADAPTER_SPECS)"
```

`unisim` has no dependency on UniLab, Hydra, or training components. MuJoCo,
Motrix, Drake, MJWarp, Genesis, IsaacGym, and IsaacSim use one public contract.
Missing proprietary SDKs or GPU workers produce an explicit cold-path
diagnostic; no backend silently falls back to another engine.

During migration, existing tasks may continue importing `unilab.base.backend`.
New consumers should use `unilab.base.backend.unisim_bridge` or import `unisim`
directly. The compatibility layer is temporary and is removed by roadmap #1428
Child 12; do not add backend-specific APIs to it.

Benchmark v1 reserves only `BenchmarkCase`, `BenchmarkResult`, and provenance
schema. Workloads, timing, comparisons, and performance claims require a
separately authorized issue.
