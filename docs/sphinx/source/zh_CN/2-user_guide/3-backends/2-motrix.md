# Motrix 后端

Motrix 是一个可选后端，通过 `motrix` extra 安装。该 extra 委托给
`unisim-core[motrix]`，runtime 版本固定在 UniSim 的 `pyproject.toml`
中，适配层位于 `unisim.backend.motrix` 下。

Motrix 是 CPU-authoritative packed HOST_BRIDGE 后端。issue #2054 将其恢复
进 tensor-only Manager runtime，但范围仅限两个 canonical workload：

```bash
uv run --extra motrix train --algo sac --task g1_walk_flat --sim motrix
uv run --extra motrix train --algo flashsac --task g1_motion_tracking --sim motrix
```

其他 Motrix owner 仍不在支持范围内，也不构成生产支持声明。

public tensor lifecycle 提供：

- persistent packed control D2H；
- persistent packed selected-reset D2H；
- persistent packed full state/sensor H2D；
- persistent packed selected-row reset publication H2D。

每个 phase 都可通过 transfer plan 的 diagnostic counter 计数。没有隐藏的
task-side NumPy reset composer；unsupported reset randomization 和 fixed
variants 一律 fail closed。

## 安装

```bash
uv sync --extra motrix
```

`make setup` 会执行相同的依赖同步，并安装 shell 自动补全。

## 何时使用

- workload 是上述两个 canonical owner 之一。
- 需要 CPU-authoritative packed HOST_BRIDGE 对比/参考路径。
- 所生成的支持矩阵将你的 entrypoint/task/backend 组合标记为 configured 或 tested。

## 命令

使用 `--render-mode record` 进行无头的仅视频回放。后端选择请保留在
`--sim motrix` 中，而不要单独 override `training.sim_backend`。

## 性能边界

MotrixSim native CPU physics 比 mjbatch 慢，也不要求完全对标。issue #2054
在 canonical control cycle 的实测中，Motrix 约为 MuJoCo/mjbatch 的 53%；
显式 D2H/H2D 总计约 0.1 ms，剩余差距几乎全部来自 native `step_n`。unisim
#340 已移除 selected read 中可避免的 full-batch 计算。

## CPU 亲和性

`training.dp_collector_cpu_ids` 可以为每个 collector rank 提供一个显式
CPU-id 列表。该列表以 `EnvCfg.cpu_ids` 传入环境，并在冷路径上校验：
条目必须非空、去重、非负且对所属进程可用。随后 Motrix 适配层在首次
模型加载之前将 MotrixSim 的共享 worker 线程池绑定到这些核上 —— worker
`i` 使用 `cpu_ids[i % len(cpu_ids)]` —— 环境构建同时把所属进程限制在
同一 CPU 块内，使 step 后的宿主机计算留在该 rank 的分区内。默认值
`null` 保持 MotrixSim 默认线程池策略和操作系统调度不变。校验后的 CPU
块可通过后端只读属性 `cpu_ids` 查询。
