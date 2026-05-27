# SAC

语言: 简体中文

SAC 是 UniLab 的默认 off-policy 入口之一。

## 默认入口

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

## 适用场景

- off-policy 训练
- 需要和 TD3、FlashSAC 共用同一套 off-policy 训练栈

## 运行模型

- SAC、TD3、FlashSAC 共用 `scripts/train_offpolicy.py`
- CPU 仿真和 GPU 学习通过 shared memory 解耦
- 默认是同步采集；`training.no_sync_collection=true` 可切到异步采集

## 常见命令

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

异步采集：

```bash
uv run scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco \
  training.no_sync_collection=true
```

## 关联脚本

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

## 关键字段

- `training.no_sync_collection`：打开异步采集
- `training.env_steps_per_sync`：同步模式下每轮采集步数
- `algo.num_envs`：并行环境数
- `algo.max_iterations`：训练轮数
- `training.use_amp`：支持时启用混合精度

同步模式更直观；异步模式更偏吞吐优先，适合拿同一 owner 做采样链路对照。

回放：

```bash
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco \
  training.play_only=true \
  algo.load_run="2024-02-04_12-00-00"
```

## 日志根目录

```text
logs/fast_sac/<task>/
```

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [APPO](03-appo.md)
- Next: [TD3](05-td3.md)
