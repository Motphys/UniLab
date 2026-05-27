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

## 关联脚本

```bash
uv run scripts/train_offpolicy.py algo=flashsac task=flashsac/g1_walk_flat/mujoco
```

## 常见命令

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco

uv run scripts/train_offpolicy.py \
  algo=flashsac \
  task=flashsac/g1_walk_flat/mujoco \
  training.no_play=true

uv run scripts/train_offpolicy.py \
  algo=flashsac \
  task=flashsac/g1_walk_flat/mujoco \
  training.play_only=true \
  algo.load_run="2026-04-23_14-06-57_mujoco"
```

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

- Index: [Documentation](../../../index.md)
- Previous: [TD3](05-td3.md)
