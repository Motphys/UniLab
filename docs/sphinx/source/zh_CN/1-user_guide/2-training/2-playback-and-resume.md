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

# Off-policy 回放可跳过 ONNX 导出, 仍然录制 MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
  --render-mode record training.export_onnx=false
```

`training.export_onnx=false` 目前只适用于 off-policy 回放链路, 即
`scripts/train_offpolicy.py` 以及统一 CLI 的 `--algo sac|td3|flashsac`。它会跳过
`policy.onnx` 导出与校验, 但不会影响 playback 或 MP4 录制。

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

## 回放命令边界

用户侧回放优先使用 `eval --load-run`，不要把脚本级路由 override 当作回放命令。需要关闭窗口或录制视频时，通过 `--render-mode` 表达：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1 --render-mode none
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1 --render-mode record
```

## 渲染模式

- `auto`：MuJoCo 默认导出视频；Motrix 默认打开交互窗口
- `record`：只录视频
- `none`：不回放
- `interactive`：适合 Motrix 原生交互渲染

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [统一 CLI](1-unified-cli.md)
- Next: [Hydra 覆盖规则](3-hydra-overrides.md)
