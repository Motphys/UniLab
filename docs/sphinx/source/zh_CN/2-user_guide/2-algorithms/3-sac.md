# SAC

SAC 通过共享的 off-policy 入口 `scripts/train_offpolicy.py` 选择，TD3 与 FlashSAC
也共用该脚本。主配置为 `conf/offpolicy/config.yaml`，SAC 算法的默认值位于
`conf/offpolicy/algo/sac.yaml`。当前的日志名称为 `fast_sac`。

## 运行模型

off-policy runner 通过 shared memory 把 CPU 仿真与 GPU 学习解耦：collector 子进程
填充驻留在 CPU 上的 replay buffer，learner 在 GPU 上训练。

单 GPU CUDA 或 Apple MPS 训练可选择
`training.replay_pipeline=gpu_resident`：CPU replay buffer 仍是权威数据源，learner
另外维护完整的 device mirror 并在 device 上随机采样。该模式会额外占用一份完整
replay 的 device 内存；默认仍是 `cpu_pinned_double_buffer`。MPS 路径由 learner
线程提交 mirror copy 和采样，不使用后台 Metal command submission。

默认 FastSAC learner 也是当前已验证的 replay-buffer 多 GPU SAC 实现。多卡模式通过
`training.num_gpus > 1` 打开，host 侧并行打包并分发 batch，多张 GPU 上的 learner
默认使用 `training.multi_gpu_sync_mode=local_sgd` 做 delayed-sync 参数平均。custom
SAC runtime 必须显式声明 distributed learner contract 后才能使用这条路径。完整命令、
严格同步回退和限制见 {doc}`../1-training/4-multi_gpu`。

## 快速开始

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.replay_pipeline=gpu_resident
```

两卡 MuJoCo 训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.use_symmetry=false
```

## 关键字段

对于 off-policy 回放路径（`scripts/train_offpolicy.py` / CLI `--algo sac`），设置
`training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。参
见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` 是每个 learner rank 每次 update 的 batch；多卡时全局
  update batch 为 `algo.batch_size * training.num_gpus`。
- `algo.max_iterations=500`
- 共享 off-policy 配置中的 `training.use_amp=true`
- 多 GPU SAC 使用 `training.num_gpus=<N>`；当前不支持 `algo.use_symmetry=true`。
- 多 GPU SAC 默认 `training.multi_gpu_sync_mode=local_sgd`，
  `training.multi_gpu_sync_interval=1`。

单 GPU SAC 与 TD3 可以在 off-policy double-buffer 路径下使用
`training.no_sync_collection=true`。多 GPU SAC 与 FlashSAC-B 仍要求同步采集。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
