# 多 GPU

当前已验证的多 GPU 训练路径是 SAC 的 replay-buffer 模式。入口仍然是统一 CLI：
`uv run train --algo sac ...`，多卡由共享 off-policy 配置字段
`training.num_gpus` 打开。

多 GPU runner 保持算法与 IPC 隔离：collector 在 host 侧填充 CPU replay buffer，
runner 根据各 learner rank 的请求打包 batch，并通过 pinned-memory pipeline 并行分
发到多张 GPU；各 GPU learner 使用分布式平均梯度更新。

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
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

选择非相邻物理卡时，用 `CUDA_VISIBLE_DEVICES` 映射本次运行可见的卡。例如使用物理
卡 0 和 7：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  algo.obs_normalization=false \
  algo.use_symmetry=false
```

如果只想做短冒烟验证，可以缩小迭代和环境数，并跳过训练后回放：

```bash
CUDA_VISIBLE_DEVICES=0,7 uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
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
代数、回放设置、logger 和可见 GPU 一致，并优先比较稳定阶段的
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
