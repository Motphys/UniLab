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

Backend physics is now owned exclusively by `unisim-core`. UniLab keeps only
the owner-layer assembly entry point `unilab.base.backend_factory`; contracts
and adapters are imported from `unisim`. The former `unilab.base.backend`
implementation and compatibility layer have been removed; do not add backend
APIs to UniLab.

Benchmark v1 reserves only `BenchmarkCase`, `BenchmarkResult`, and provenance
schema. Workloads, timing, comparisons, and performance claims require a
separately authorized issue.
