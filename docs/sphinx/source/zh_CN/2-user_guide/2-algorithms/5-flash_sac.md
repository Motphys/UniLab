# FlashSAC

语言: 简体中文

FlashSAC 是共享 off-policy 入口上的第三个算法。使用 `--algo flashsac` 选择它；默认
值位于 `conf/offpolicy/algo/flashsac.yaml`，实现位于
`src/unilab/algos/torch/flash_sac/` 下。

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
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

`scripts/train_offpolicy.py` 会拒绝 FlashSAC 的 `training.num_gpus > 1`，因此除非实
现发生变化，否则请保持默认的单 GPU 路径。

## Navigation

- Index: [文档](0-index.md)
