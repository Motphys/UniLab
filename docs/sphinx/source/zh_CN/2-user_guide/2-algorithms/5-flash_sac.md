# FlashSAC

FlashSAC 通过 `src/unilab/scripts/train_flashsac.py` 运行，拥有独立的配置树。使用
`--algo flashsac` 选择它；默认值内联在 `src/unilab/conf/flashsac/config.yaml` 中，实现位于
`uni_rl.algos.flash_sac` (unilab-rl repo) 下。

它与 SAC、TD3 共用 off-policy runner 设计，但默认网络并不相同：actor 使用
block-based 结构，critic 使用 distributional（categorical）Q 变体。

## 快速开始

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## 关键字段

对于 off-policy 回放路径（`src/unilab/scripts/train_flashsac.py` / CLI `--algo flashsac`），设
置 `training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。
参见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

FlashSAC 要求同步采集，并与 SAC、TD3 共用唯一 replay 路径：有界 host ingress 加
一个驻留在 CUDA 或 Apple MPS learner device 上的完整 replay ring。CPU 与 XPU
training 不受支持。

日志根目录为 `logs/flash_sac/<task>/`。

## 多卡数据并行

FlashSAC 与 SAC 共用同一套多卡数据并行机制：`training.devices` 下每个 rank 各跑一
套独立的 learner+collector；启动时广播完整模型状态，稳态在每个实际 optimizer step
前分别平均 actor / critic / temperature 梯度。仅 rank 0 保存 checkpoint。用法与限制见
{doc}`/zh_CN/2-user_guide/2-algorithms/3-sac` 的"多卡数据并行"小节。

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  training.devices=[0,1]
```
