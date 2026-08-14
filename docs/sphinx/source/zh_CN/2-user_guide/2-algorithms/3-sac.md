# SAC

SAC 通过共享的 off-policy 入口 `scripts/train_offpolicy.py` 选择，TD3 与 FlashSAC
也共用该脚本。主配置为 `conf/offpolicy/config.yaml`，SAC 算法的默认值位于
`conf/offpolicy/algo/sac.yaml`。当前的日志名称为 `fast_sac`。

## 运行模型

off-policy runner 通过有界 shared memory 把 CPU 仿真与 accelerator 学习解耦。
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

对于 off-policy 回放路径（`scripts/train_offpolicy.py` / CLI `--algo sac`），设置
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
`training.dp_collector_cpu_ids` 显式指定。

当前限制：

- 仅 SAC 与 FlashSAC：TD3 learner 未实现 optimizer-boundary gradient sync contract，
  多卡启动即报错。
- NCCL gradient collective 不捕获进 optimizer CUDA Graph；若多卡配置请求 actor/critic
  graph capture，learner 会自动切换到 eager optimizer update 并继续同步梯度，单卡行为不变。
  runtime manifest 的 `dp_sync.cuda_graph_optimizer_capture=eager_fallback` 会记录该回退。
- 仅验证过 `mujoco` backend。
- 仅单节点：rank 之间通过 run 目录里的 FileStore rendezvous，NCCL 走 TCP
  loopback（默认 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1`，环境变量显式设置
  时优先）——部分机型（如 RTX 6000D）的 NCCL P2P/SHM peer transport 不可靠，
  TCP loopback 是唯一稳定传输。

collector `Steps/s` 与 learner `Samples/s` 各自相对单卡的 scaling 基准见
`benchmark/rl/benchmark_offpolicy_dp_scaling.py`（issue #968，真实运行不进 CI）。
