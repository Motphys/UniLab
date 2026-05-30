# 续训与检查点

语言: 简体中文

检查点选择由算法层面的字段控制。请使用 `algo.load_run`，而不是
`training.load_run`。

## 续训

使用 run id，或用 `-1` 表示相关日志目录中最新的一次运行：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true

uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.load_run=2026-03-16_01-35-12_mujoco \
  training.no_play=true
```

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

## Navigation

- Index: [文档](0-index.md)
