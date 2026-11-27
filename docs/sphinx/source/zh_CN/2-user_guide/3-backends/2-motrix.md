# Motrix 后端

## MotrixSim GPU

安装 `mtxwarp` extra 并指定 `--sim motrix_gpu`，即可让 MotrixSim 物理仿真
和 Manager 张量使用同一张 NVIDIA GPU。该配置使用 MotrixSim 1.0.0 与 Warp
1.17.0，支持 Linux x86_64。

该 extra 从已发布的 UniSim 2.9.0 wheel 安装适配器。

```bash
uv run --locked --extra mtxwarp train --algo appo --task go2_joystick_flat --sim motrix_gpu
uv run --locked --extra mtxwarp train --algo sac --task g1_walk_flat --sim motrix_gpu
uv run --locked --extra mtxwarp train --algo flashsac --task g1_motion_tracking --sim motrix_gpu
```

各 task owner 定义每个仿真环境的接触和约束容量。G1 Walk 使用即时执行的
选行重置，以保留宿主机上的课程更新。其余两个 owner 启用 CUDA Graph 自动
重置。调试时可以用 `env.autoreset_cuda_graph=false` 切换为即时执行。
G1 动作跟踪 owner 禁用了当前后端不支持的质心随机化。引擎支持范围见
UniSim 支持矩阵。

## MotrixSim CPU

Motrix 是一个可选后端，通过 `motrix` extra 安装。该 extra 委托给
`unisim-core[motrix]`，runtime 版本固定在 UniSim 的 `pyproject.toml`
中，适配层位于 `unisim.backend.motrix` 下。

Motrix 是 CPU-authoritative packed HOST_BRIDGE 后端。issue #2054 将其恢复
进 tensor-only Manager runtime，当前包含这些 canonical workload：

```bash
uv run --extra motrix train --algo appo --task go2_joystick_flat --sim motrix
uv run --extra motrix train --algo sac --task g1_walk_flat --sim motrix
uv run --extra motrix train --algo flashsac --task g1_motion_tracking --sim motrix
```

其他 Motrix owner 仍不在支持范围内，也不构成生产支持声明。
Go2 APPO owner 是已恢复的 configured 训练路径；当前证据仅限 bounded smoke
training，不构成性能 benchmark。

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

- workload 是上述 canonical owner 之一。
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
