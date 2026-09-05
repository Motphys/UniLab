# PPO

PPO 是默认的同步 on-policy 训练路径。它使用 `src/unilab/scripts/train_rsl_rl.py`，从
`src/unilab/conf/ppo/config.yaml` 组合配置，并运行 `uni_rl.algos.rsl_rl_ppo` (unilab-rl repo)
和 `src/unilab/training/rsl_rl.py` 中的 RSL-RL 适配代码。

## 快速开始

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## 常用 Override

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

使用 `uv run eval` 进行检查点回放：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

日志按 `algo.algo_log_name` 分组；`src/unilab/conf/ppo/config.yaml` 中的默认值为
`rsl_rl_ppo`。

## 单机多卡训练

`training.devices` 打开 RSL-RL 的同步数据并行 PPO：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  'training.devices=[0,1]' \
  training.no_play=true

uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  'training.devices=[0,1]' \
  training.no_play=true
```

`null` 或 `[]` 保持自动选择单个 device；`[d]` 显式选择一张 CUDA 卡；两个以上
索引会按列表顺序在本机为每张卡启动一个进程。不能同时设置 `training.device` 与
`training.devices`。父进程已有 `CUDA_VISIBLE_DEVICES` 时，配置索引仍按父进程可见
设备解释，并保留用户给定顺序。

对 IsaacGym、IsaacSim 和 Genesis owner，同一拓扑也会传给环境仿真器。torchrun
worker 继承重映射后的 `CUDA_VISIBLE_DEVICES`，因此传给 worker 的是本地索引（例如
worker 看到 `[4,5]` 时，主机设备 5 传为 `device_id=1`）；off-policy worker 保持父进程
可见设备索引。Genesis 会在 `gs.init` 前选择每个进程的 session 设备。

`algo.num_envs` 是**每个 rank** 的环境数，不是全局预算。设 rank 数为 `W`、配置
环境数为 `N`、rollout 长度为 `T`：

```text
每 rank 每轮样本数 = N * T
全局每轮样本数     = W * N * T
```

每个 rank 独立持有 env、policy copy 和 rollout storage；rollout、GAE、advantage
normalization 与 mini-batch shuffle 都留在本地。RSL-RL 在启动时广播 rank 0 的模型
状态，并在每个 PPO mini-batch backward 后平均梯度。rank `i` 使用
`algo.seed + i`。`training.num_timesteps` 按全局样本预算解释，因此推导 iteration
数时使用 `W * N * T`。

只有 rank 0 写 TensorBoard/W&B、checkpoint、`run_config.json` 与
`run_summary.json`；所有 rank 销毁 process group 后，才由 rank 0 可选进入
playback。`run_summary.json` 记录 world size、每 rank/全局环境数、每轮样本数和
聚合训练吞吐率。RSL-RL 的 `Perf/total_fps` 同样是全局 samples/s；reward 与
episode 统计仍是 rank 0 的本地视角。

RSL-RL 启动后不会同步 observation normalizer buffer 或环境 curriculum 状态。因此，
启用 empirical normalization 的 task（包括当前 Go2 flat owner）保留 rank-local
统计，checkpoint 保存 rank 0 的副本。这是上游 RSL-RL 的分布式语义，不是先拼接
全局 rollout 再训练的 PPO。

集成 launcher 只覆盖单机。仓库已有 `go2_joystick_flat` 与
`g1_motion_tracking` 的双卡 MuJoCo smoke；这不构成多机或 Motrix 多卡 support
声明。当前验证的 RTX 6000D 主机沿用 production NCCL 兼容默认值
`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`，用户显式环境变量优先。
