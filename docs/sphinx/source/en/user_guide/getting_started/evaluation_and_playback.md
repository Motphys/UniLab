# Evaluation and Playback

```bash
# Latest run
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Headless video export
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --render-mode record

# Off-policy playback can skip ONNX export and still record MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
    --render-mode record training.export_onnx=false

# Demo (uses a baked-in checkpoint)
uv run demo
```

Render modes:

- `interactive` — open viewer window (default on macOS Motrix).
- `record` — write MP4 to `runs/<run>/playback/`.
- `none` — skip rendering, just compute metrics.

`training.export_onnx=false` currently applies only to the off-policy playback path
(`scripts/train_offpolicy.py` and CLI runs with `--algo sac|td3|flashsac`). It skips
`policy.onnx` export and verification but still runs playback and video recording.

See `unilab.visualization.playback` for the underlying API.
