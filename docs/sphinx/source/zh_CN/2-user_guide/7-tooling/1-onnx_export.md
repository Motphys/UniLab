# ONNX 导出

ONNX 导出与训练脚本中的回放绑定。PPO 和 HIM-PPO 脚本在作为脚本运行时会设置 `EXPORT_POLICY=True`，然后在 `training.play_only=true` 回放期间导出。APPO、off-policy 和 MLX 回放路径也会在它们的脚本代码中导出 `policy.onnx`，并用 ONNX Runtime 验证它。

## 示例

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

使用与生成 checkpoint 时相同的 `--algo`、`--task` 和 `--sim` 值。关于部署上下文，参见 {doc}`../../3-deployment/1-sim_to_real/5-onnx_runtime`。
