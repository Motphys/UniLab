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

SAC 与 TD3 可以在 off-policy device replay 路径下使用
`training.no_sync_collection=true`。FlashSAC-B 要求同步采集。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
