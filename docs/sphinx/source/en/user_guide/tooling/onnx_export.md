# ONNX Export

ONNX export is tied to playback in the training scripts. The PPO and HIM-PPO
scripts set `EXPORT_POLICY=True` when run as scripts, then export during
`training.play_only=true` playback. APPO, off-policy, and MLX playback paths
also export `policy.onnx` and verify it with ONNX Runtime in their script code.

## Examples

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1

uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

Use the same task owner and algorithm family that produced the checkpoint. For
deployment context, see {doc}`../../deployment/sim_to_real/onnx_export_and_runtime`.
