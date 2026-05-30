# 评估与回放

语言: 简体中文

```bash
# 最近一次运行
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# 无头视频导出
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --render-mode record

# Off-policy 回放可以跳过 ONNX 导出，但仍然录制 MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
    --render-mode record training.export_onnx=false

# 演示（使用内置的检查点）
uv run demo
```

渲染模式：

- `interactive` — 打开查看器窗口（macOS Motrix 上的默认值）。
- `record` — 将 MP4 写入 `runs/<run>/playback/`。
- `none` — 跳过渲染，仅计算指标。

`training.export_onnx=false` 目前仅适用于 off-policy 回放路径
（`scripts/train_offpolicy.py` 以及使用 `--algo sac|td3|flashsac` 的 CLI 运行）。它会跳过
`policy.onnx` 的导出与校验，但仍会执行回放和视频录制。

底层 API 请参阅 `unilab.visualization.playback`。

## Navigation

- Index: [文档](0-index.md)
