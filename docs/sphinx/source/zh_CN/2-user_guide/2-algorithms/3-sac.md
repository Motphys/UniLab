# SAC

SAC 通过共享的 off-policy 入口 `scripts/train_offpolicy.py` 选择，TD3 与 FlashSAC
也共用该脚本。主配置为 `conf/offpolicy/config.yaml`，SAC 算法的默认值位于
`conf/offpolicy/algo/sac.yaml`。当前的日志名称为 `fast_sac`。

## 运行模型

off-policy runner 通过 shared memory 把 CPU 仿真与 GPU 学习解耦：collector 子进程
填充驻留在 CPU 上的 replay buffer，learner 在 GPU 上训练。

SAC 也是当前已验证的 replay-buffer 多 GPU 训练算法。多卡模式通过
`training.num_gpus > 1` 打开，host 侧并行打包并分发 batch，多张 GPU 上的 learner
进行平均梯度更新。完整命令与限制见 {doc}`../1-training/4-multi_gpu`。

## 快速开始

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_rough --sim motrix training.no_play=true
```

两卡 MuJoCo 训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

## 关键字段

对于 off-policy 回放路径（`scripts/train_offpolicy.py` / CLI `--algo sac`），设置
`training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。参
见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=fast_sac`
- `algo.num_envs=4096`
- `algo.max_iterations=500`
- 共享 off-policy 配置中的 `training.use_amp=true`
- 多 GPU SAC 使用 `training.num_gpus=<N>`；当前验证要求
  `algo.obs_normalization=false`，且不支持 `algo.use_symmetry=true`。

`scripts/train_offpolicy.py` 中当前的 runner 路径要求同步采集；脚本会拒绝
`training.no_sync_collection=true`。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```
