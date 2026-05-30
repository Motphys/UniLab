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
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.no_sync_collection=true
```

## 实现脚本

SAC 顶层 CLI 会路由到 `scripts/train_offpolicy.py`，并选择
`conf/offpolicy/task/sac/` 下的 owner YAML。普通调参继续用 Hydra override；
算法、任务和后端选择放在 `--algo`、`--task`、`--sim`。

## 关键字段

- `training.no_sync_collection`：打开异步采集
- `training.env_steps_per_sync`：同步模式下每轮采集步数
- `algo.num_envs`：并行环境数
- `algo.max_iterations`：训练轮数
- `training.use_amp`：支持时启用混合精度

同步模式更直观；异步模式更偏吞吐优先，适合拿同一 owner 做采样链路对照。

回放：

```bash
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco \
  --load-run "2024-02-04_12-00-00"
uv run eval --algo sac --task g1_walk_flat --sim mujoco \
  --load-run -1 --render-mode record training.export_onnx=false
```

当你只想回放 / 录制 MP4, 不想先导出 `policy.onnx` 时, 可以设置
`training.export_onnx=false`。这个开关只在 off-policy 回放链路生效；通用说明见
[评估、回放与恢复训练](../2-training/2-playback-and-resume.md)。

## 日志根目录

```text
logs/fast_sac/<task>/
```

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [APPO](3-appo.md)
- Next: [TD3](5-td3.md)
