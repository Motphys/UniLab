# FlashSAC

FlashSAC 是共享 off-policy 入口上的第三个算法。使用 `--algo flashsac` 选择它；默认
值位于 `conf/offpolicy/algo/flashsac.yaml`，实现位于
`src/unilab/algos/torch/flash_sac/` 下。

它与 SAC、TD3 共用 off-policy 训练脚本，但默认网络并不相同：actor 使用 block-based
结构，critic 使用 distributional（categorical）Q 变体。

## 快速开始

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## 关键字段

对于 off-policy 回放路径（`scripts/train_offpolicy.py` / CLI `--algo flashsac`），设
置 `training.export_onnx=false` 可在仍然录制回放视频的同时跳过 `policy.onnx` 导出。
参见 {doc}`/zh_CN/1-getting_started/3-evaluation_and_playback`。

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

FlashSAC learner 当前没有声明分布式 learner contract，因此
`training.num_gpus > 1` 会被 capability check 拒绝。除非 FlashSAC 实现并验证多
GPU 同步语义，否则请保持默认的单 GPU 路径。

日志根目录为 `logs/flash_sac/<task>/`。
