# UniSim Core 迁移

UniLab 的物理后端正在迁移到独立的 `unisim-core` package。安装命令和
Python namespace 分别是：

```bash
uv sync --extra mujoco
uv run python -c "import unisim; print(unisim.ADAPTER_SPECS)"
```

`unisim` 不依赖 UniLab、Hydra 或训练组件；MuJoCo、Motrix、Drake、MJWarp、
Genesis、IsaacGym 和 IsaacSim 都通过同一个公开 contract 暴露。专有 SDK 或
GPU worker 缺失时，构造 backend 会在冷路径给出明确诊断，不会静默回退到
另一引擎。

迁移期间旧的 `unilab.base.backend` 路径仍可用于已有 task。新代码应使用
`unilab.base.backend.unisim_bridge` 或直接导入 `unisim`。该兼容层是临时的，
在 roadmap #1428 Child 12 中删除；不要在其上新增 backend-specific API。

benchmark v1 目前只保留 `BenchmarkCase`、`BenchmarkResult` 和 provenance
schema，不包含 workload、计时或性能结论。
