# FlashSAC

语言: 简体中文

FlashSAC 是共享 off-policy 入口的另一条 owner 路径，但网络结构和默认超参数与 FastSAC 不同。

## 默认入口

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
```

## 适用场景

- FlashSAC owner 路径
- 共享 off-policy 训练脚本

## 网络与运行模型

- 仍走 shared-memory off-policy runner
- actor 使用 block-based 结构，critic 使用 distributional Q 变体
- 和 FastSAC 共用训练脚本，但不是同一套默认网络

## 实现脚本

FlashSAC 顶层 CLI 会路由到 `scripts/train_offpolicy.py`，并选择
`conf/offpolicy/task/flashsac/` 下的 owner YAML。普通调参继续用 Hydra
override；算法、任务和后端选择放在 `--algo`、`--task`、`--sim`。

## 常见命令

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco

uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  training.no_play=true

uv run eval --algo flashsac --task g1_walk_flat --sim mujoco \
  --load-run "2026-04-23_14-06-57_mujoco"
uv run eval --algo flashsac --task g1_walk_flat --sim mujoco \
  --load-run -1 --render-mode record training.export_onnx=false
```

如果只需要 off-policy 回放或录制 MP4, 可以设置 `training.export_onnx=false`
跳过 `policy.onnx` 导出。该开关只在 `train_offpolicy.py` 链路生效；范围与统一
CLI 用法见 [评估、回放与恢复训练](../2-training/2-playback-and-resume.md)。

## 关键字段

- `algo.max_iterations`
- `algo.num_envs`
- `algo.tau`
- `algo.save_interval`
- `algo.algo_params.actor_num_blocks`
- `algo.algo_params.critic_num_blocks`

它和 FastSAC 共用脚本入口，但 actor / critic 结构不是同一套默认网络。

## 日志根目录

```text
logs/flash_sac/<task>/
```

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [TD3](5-td3.md)
