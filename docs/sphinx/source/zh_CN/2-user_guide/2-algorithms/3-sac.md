# SAC

语言: 简体中文

SAC 通过共享的 off-policy 入口 `scripts/train_offpolicy.py` 选择。主配置为
`conf/offpolicy/config.yaml`，SAC 算法的默认值位于 `conf/offpolicy/algo/sac.yaml`。
当前的日志名称为 `fast_sac`。

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
- `algo.max_iterations=500`
- 共享 off-policy 配置中的 `training.use_amp=true`

`scripts/train_offpolicy.py` 中当前的 runner 路径要求同步采集；脚本会拒绝
`training.no_sync_collection=true`。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```

## Navigation

- Index: [文档](0-index.md)
