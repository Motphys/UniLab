# Evaluation and Playback

```bash
# Latest run
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Headless video export
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --render-mode record

# Demo (uses a baked-in checkpoint)
uv run demo
```

Render modes:

- `interactive` — open viewer window (default on macOS Motrix).
- `record` — write MP4 to `runs/<run>/playback/`.
- `none` — skip rendering, just compute metrics.

See `unilab.visualization.playback` for the underlying API.
