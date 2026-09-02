# SAC

SAC 通过 `src/unilab/scripts/train_sac.py` 运行；TD3 与 FlashSAC 各有独立的入口与按算法
划分的配置树。主配置为 `src/unilab/conf/sac/config.yaml`，SAC 算法的默认值内联在其中。
当前的日志名称为 `fast_sac`。

## 运行模型

off-policy runner 通过有界 shared memory 把仿真采集与 accelerator 学习解耦。
collector 子进程通过两个 packed ingress slot 发布 transition，完整 replay ring 只由
一个 CUDA 或 Apple MPS learner device 持有。因此 host replay 分配不随 replay
capacity 增长；slot 的 device copy 完成后才推进 `ptr` 与 `size`。CUDA 通过 side
stream 提交和 commit，MPS 的 device work 只由 learner 线程提交，不使用后台 Metal
submission。CPU 与 XPU training 不受支持，也不存在第二套 replay pipeline。

## 快速开始

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
```

## 关键字段

对于 off-policy 回放路径（`src/unilab/scripts/train_sac.py` / CLI `--algo sac`），设置
`training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。参
见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` 是 learner 每次 update 的 batch。
- `algo.max_iterations=500`
- 共享 off-policy 配置中的 `training.use_amp=true`

off-policy device replay 路径统一使用同步、learner-owned inference：collector 通过
shared memory 交换 observation/action，不持有 actor。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```

## 多卡数据并行

`training.devices` 打开单节点多卡数据并行（data parallel）：rank i 在
`cuda:devices[i]` 上各跑一套独立的 learner+collector，rank 0 负责 spawn 其余
rank 子进程。启动时 rank 0 一次性广播 actor、critic、target critic 和温度状态；稳态
训练不再平均参数，而是在每个实际 optimizer step 前对对应的 actor、critic 或温度梯度
执行阻塞式 flat-gradient `all_reduce(SUM) / world_size`。各 rank 不交换 replay 数据，
在相同初值、平均梯度和更新顺序下各自维护一致的 optimizer 状态。

每个 rank 当前只创建一个 collector。使用 mjwarp 时，rank i 会在 probe env 和 collector
正式 env materialization 之前，把 Warp 的进程默认/当前 device 显式绑定到该 rank 的
learner device `cuda:devices[i]`；因此 collector 不依赖 Warp 新进程默认的 `cuda:0`，也不
会跨 rank 集中到同一张卡。runtime manifest 的 `collector_backend_device` 记录本 rank 的
实际绑定。

off-policy 只公开 `training.devices` 这一个设备字段：`null` 或 `[]` 自动选择单个
learner device，`[0]` 显式选择 `cuda:0`，两个以上索引才启动多卡拓扑。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.devices=[0,1]
```

rank 0 独占终端与 TensorBoard/W&B logger；其他 learner 不刷新终端，也不创建独立
tfevents。rank 0 在每个 learner iteration 汇总跨 rank 标量：loss、reward 和 timing
取均值，计数与并行吞吐求和。`perf/steps_per_sec` / 终端 `Steps/s` 表示所有
collector 的聚合 env-step 吞吐，`perf/effective_samples_per_sec` / 终端 `Samples/s`
表示所有 learner 的聚合有效样本吞吐。checkpoint 同样由 rank 0 独占：每个保存间隔和
训练结束只在 canonical run 目录写一份模型；其他 rank 复用该路径完成进程协调，不创建
rank 子目录或任何日志文件。
自动生成的多卡 run 目录以 `_gpuxN` 结尾（例如 `_gpux2`）；单卡目录和显式
`training.log_dir` 保持原样。

collector 的 CPU 亲和按 rank 自动均分（`cpu_count // world_size` 一段），可用
`training.dp_collector_cpu_ids` 显式指定。该核区经 `EnvCfg.cpu_ids` 生效：除
MuJoCo worker 线程逐核绑定外，collector 进程本身（含 Numba 并行 kernel 线程池，池
大小取核区长度）也被限制在同一核区内，避免跨 rank 抢占。

当前限制：

- 仅 SAC 与 FlashSAC：TD3 learner 未实现 optimizer-boundary gradient sync contract，
  多卡启动即报错。
- 多卡 actor/critic optimizer CUDA Graph 支持捕获 NCCL gradient all-reduce。首次 capture
  前，DP owner 会先执行一次 eager all-reduce 并同步设备，完成 NCCL collective lazy
  initialization；flat-gradient buffer 在 graph 生命周期内保持固定地址。runtime manifest
  记录 `dp_sync.cuda_graph_optimizer_capture=enabled_after_collective_warmup` 和
  `cuda_graph_collective_warmup=true`。销毁 process group 前会先释放 graph 中的 NCCL 节点。
  Issue #978 的验证环境为 PyTorch 2.7.0+cu128、CUDA 12.8、NCCL 2.26.2、2 × RTX
  6000D；TCP loopback 下同步 all-reduce 和 `async_op=True` + `Work.wait()` 均可 capture/replay，
  默认 stream 与 side stream 均通过；跳过 warmup 时首个 all-reduce 会在 capture 中报
  `operation not permitted when stream is capturing`。有限超时的最小复现见
  `scripts/benchmark/rl/reproduce_nccl_cuda_graph_capture.py`。
- MuJoCo 有已提交的多卡 scaling benchmark；mjwarp 的 per-rank device placement 有
  `tests/base/backend/test_process_device.py` 与 off-policy runner/worker 单测覆盖，但仓库中
  尚无 mjwarp 多卡吞吐或收敛 benchmark。
- 仅单节点：rank 之间通过 run 目录里的 FileStore rendezvous，NCCL 走 TCP
  loopback（默认 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1`，环境变量显式设置
  时优先）——部分机型（如 RTX 6000D）的 NCCL P2P/SHM peer transport 不可靠，
  TCP loopback 是唯一稳定传输。

collector `Steps/s` 与 learner `Samples/s` 各自相对单卡的 scaling 基准见
`scripts/benchmark/rl/benchmark_offpolicy_dp_scaling.py`（issue #968，真实运行不进 CI）。
