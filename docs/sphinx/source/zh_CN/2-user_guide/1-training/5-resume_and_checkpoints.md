# 续训与检查点

检查点选择由算法层面的字段控制。请使用 `algo.load_run`，而不是
`training.load_run`。

## 续训

PPO/APPO 直接通过 `algo.load_run` 续训。使用 run id，或用 `-1` 表示相关日志目录中最新的一次运行：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true
```

off-policy 算法（SAC/FlashSAC/WarpSAC）的续训是显式开启的，以避免
`algo.load_run=-1` 的默认值把全新启动意外变成续训。设置 `algo.resume=true`，并用
`algo.load_run`（run id 或 `-1` 表示最新 run）与可选的 `algo.checkpoint`
（迭代号或文件名；`-1` 表示最新 checkpoint）选择检查点：

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  --profile mimiclite_dr \
  algo.resume=true \
  algo.load_run=2026-10-05_14-28-04_mjwarp \
  training.no_play=true
```

off-policy checkpoint 恢复完整的 learner 状态（网络、优化器、调度器、normalizer
与 update count），训练以绝对迭代编号继续（`model_32000.pt` 从第 32001 次迭代继续，
因此 `algo.max_iterations` 仍是总预算而不是增量）。replay buffer 不落盘：续训的 run
会从空 buffer 重新经过 train-start 阈值热身，且 RNG 状态不恢复。

## 回放检查点

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` 会将 `--load-run` 映射到底层的检查点选择器，并设置回放模式：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

某些脚本路径接受通过 `algo.load_run` 传入的检查点路径；统一 CLI 会将 `--load-run`
校验为一个 run id，且不接受路径分隔符。

## 随机种子

训练种子的解析在 `src/unilab/training/seed.py` 中实现。算法配置目前携带
`algo.seed`，并且在实验跟踪启用时，该辅助逻辑会记录种子元数据。
