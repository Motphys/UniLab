# 多 GPU

当前已验证的多 GPU 训练路径是 SAC 的 replay-buffer 模式。入口仍然是统一 CLI：
`uv run train --algo sac ...`，多卡由共享 off-policy 配置字段
`training.num_gpus` 打开。

多 GPU runner 保持算法与 IPC 隔离：collector 在 host 侧填充 CPU replay buffer，
runner 根据各 learner rank 的请求打包 batch，并通过 pinned-memory pipeline 并行分
发到多张 GPU。collector 仍是单进程 CPU collector；多卡不会隐式增加
`algo.num_envs`。

## Learner 同步语义

SAC 多 GPU 默认使用 `training.multi_gpu_sync_mode=local_sgd`。每个 learner rank
在自己的 GPU 上独立做本地 SAC update，不在每个 critic / actor update 后同步梯度；
runner 在 iteration 边界按 `training.multi_gpu_sync_interval` 对 actor、critic、
target critic 和 entropy coefficient 参数做一次跨 rank 平均，然后 rank0 在该同步
边界把 actor 权重发布给 CPU collector。默认 `training.multi_gpu_sync_interval=1`，表示每个
learner iteration 同步一次；增大该值可以进一步减少 4 卡、8 卡时的通信频率，但会
增加 rank 间参数漂移。`local_sgd` 下 optimizer state 按设计保持 rank-local，不在同
步边界平均，以避免把通信量扩展到 AdamW 动量状态。

如需严格的每次 update 梯度平均，可显式设置
`training.multi_gpu_sync_mode=sync_sgd`。该模式更接近单卡 global batch 的同步
SGD 语义，但通信次数更多，通常不适合没有高速 GPU 互联的机器。

## Batch 语义

`algo.batch_size` 在 off-policy SAC 中定义为**每个 learner rank 每次 update 的
batch**，不是跨所有 GPU 的 global batch。`training.num_gpus=N` 时，每次 update 的
全局有效 batch 为 `algo.batch_size * N`；每个 rank 会从共享 replay buffer 独立随机
采样自己的 batch。默认 `local_sgd` 模式下这些本地 update 的参数在 iteration 边界
平均；`sync_sgd` 模式下则在每次 update 后对梯度做分布式平均。

因此，两卡运行在相同 `algo.batch_size` 下会比单卡每次 update 使用两倍有效样本。
如果要保持单卡和多卡的全局 update batch 一致，需要显式把多卡运行的
`algo.batch_size` 按 GPU 数缩小，例如单卡 `algo.batch_size=8192` 对应两卡
`algo.batch_size=4096`。日志中的 `Batch/Rank` 是该 per-rank batch，`Batch/Update`
是全局 batch。

## 前置条件

- 只支持 SAC：`training.num_gpus > 1` 会拒绝 TD3、FlashSAC、PPO、MLX PPO 和 APPO。
- 必须使用 CUDA 设备；用 `CUDA_VISIBLE_DEVICES` 选择物理卡。
- 本轮验证要求 `algo.obs_normalization=false`。
- SAC 的对称增强当前不支持多卡；若任务 owner 默认开启，需要设置
  `algo.use_symmetry=false`。
- 采集必须同步；不要设置 `training.no_sync_collection=true`。

## 基本命令

两张相邻卡：

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

选择非相邻物理卡时，用 `CUDA_VISIBLE_DEVICES` 映射本次运行可见的卡。例如使用物理
卡 0 和 7：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

如果只想做短冒烟验证，可以缩小迭代和环境数，并跳过训练后回放：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd \
  algo.obs_normalization=false \
  algo.use_symmetry=false \
  algo.max_iterations=10 \
  algo.num_envs=512 \
  training.no_play=true
```

日志仍写入 SAC 的默认目录：`logs/fast_sac/<TaskName>/`。

## 性能检查

多 GPU 主要减少 learner 更新瓶颈；如果环境数、batch 或迭代数太小，分布式启动、
batch 打包和梯度同步开销可能超过收益。对比单卡和多卡时，请保持任务、环境数、迭
代数、回放设置、logger 和可见 GPU 一致；同时明确是比较相同 per-rank batch 还是相
同 global batch，并优先比较稳定阶段的
`train_fps`、learner step 时间和端到端迭代时间。

## 常见错误

- `Only SAC supports training.num_gpus > 1`：当前只验证 SAC。
- `SAC multi-GPU training requires a CUDA device`：没有可用 CUDA，或
  `training.device` 被设成了 CPU。
- `requires algo.obs_normalization=false`：显式追加
  `algo.obs_normalization=false`。
- `set training.num_gpus=1 or algo.use_symmetry=false`：多卡 SAC 暂不支持对称增
  强，显式追加 `algo.use_symmetry=false`。

修改多 GPU 行为时，请在最接近 off-policy runner 与 IPC 边界处进行验证，而不是仅检
查顶层命令。
