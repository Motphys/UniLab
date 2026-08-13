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
rank 子进程。同步是参数级的：启动时先广播 rank 0 的初始参数，之后每
`training.dp_sync_interval`（默认 8）个 learner iteration 对 actor+critic 参数做
一次 all-reduce 平均；梯度不交换，各 rank 保留自己的 optimizer 状态。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.devices=[0,1] \
  training.dp_sync_interval=8
```

每个 rank 的日志落在同一 run 目录下：rank 0 直接在 run 目录（含
`run_summary.json` 与 tfevents），rank i>0 在 `rank{i}/` 子目录（仅 tfevents）。
collector 的 CPU 亲和按 rank 自动均分（`cpu_count // world_size` 一段），可用
`training.dp_collector_cpu_ids` 显式指定。

当前限制：

- 仅 SAC：TD3 / FlashSAC 的 learner 未实现 `dp_sync_tensors()`，多卡启动即报错。
- 仅验证过 `mujoco` backend；`training.devices` 与 `training.device` 互斥。
- 仅单节点：rank 之间通过 run 目录里的 FileStore rendezvous，NCCL 走 TCP
  loopback（默认 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1`，环境变量显式设置
  时优先）——部分机型（如 RTX 6000D）的 NCCL P2P/SHM peer transport 不可靠，
  TCP loopback 是唯一稳定传输。

2 卡聚合吞吐相对单卡的 scaling 验收基准见
`benchmark/rl/benchmark_offpolicy_dp_scaling.py`（issue #968，真实运行不进 CI）。
