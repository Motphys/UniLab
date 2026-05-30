# ONNX Export

ONNX export is tied to playback in the training scripts. The PPO and HIM-PPO
scripts set `EXPORT_POLICY=True` when run as scripts, then export during
`training.play_only=true` playback. APPO, off-policy, and MLX playback paths
also export `policy.onnx` and verify it with ONNX Runtime in their script code.

## Examples

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

Use the same `--algo`, `--task`, and `--sim` values that produced the
checkpoint. For deployment context, see
{doc}`../../3-deployment/1-sim_to_real/5-onnx_runtime`.
