# 评估、回放与恢复训练

语言: 简体中文

## 回放最近一次训练

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

## 回放指定 run

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco \
  --load-run 2026-04-24_01-36-01_mujoco
```

## Motrix 回放

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

## 恢复训练

统一 CLI 恢复训练仍通过 Hydra override 表达：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco algo.load_run=-1
```

恢复指定 run：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=2026-04-24_01-36-01_mujoco
```

## 低层脚本回放

需要直接调试脚本入口时，统一用 `training.play_only=true`：

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco training.play_only=true
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco training.play_only=true
```

## 渲染模式

- `auto`：MuJoCo 默认导出视频；Motrix 默认打开交互窗口
- `record`：只录视频
- `none`：不回放
- `interactive`：适合 Motrix 原生交互渲染

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [统一 CLI](01-unified-cli.md)
- Next: [Hydra 覆盖规则](03-hydra-overrides.md)
