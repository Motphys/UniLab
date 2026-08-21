# ONNX 运行时

UniLab 从既有的训练回放路径导出 ONNX 策略。使用产出该检查点的同一算法家族与任务
owner；回放代码加载检查点、导出 `policy.onnx`，并在该路径实现了 ONNX Runtime 检查时
校验所导出的计算图。

## 导出路径

| 算法路径 | 入口脚本 | 仓库中的导出行为 |
| --- | --- | --- |
| PPO（torch） | `scripts/train_rsl_rl.py` | 脚本入口处 `EXPORT_POLICY=True`；回放调用 `runner.export_policy_to_onnx(...)` 与 `runner.export_policy_to_jit(...)`。 |
| HIM-PPO | `scripts/train_him_ppo.py` | 与 PPO 相同的脚本级导出模式。 |
| APPO | `scripts/train_appo.py` | 回放写出 `policy.onnx` 并将 ONNX Runtime 输出与 PyTorch 比对校验。 |
| SAC / TD3 / FlashSAC | `scripts/train_sac.py` / `scripts/train_td3.py` / `scripts/train_flashsac.py` | 回放写出 `policy.onnx`；SAC 与 FlashSAC 在导出前使用 `actor.as_export_module()`。 |

## 命令

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo appo --task g1_motion_tracking --sim motrix --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` 设置回放模式，并把 `--load-run` 映射到所路由训练脚本使用的检查点
选择器。导出的文件会写入所选的运行目录。对于部署原型，请把导出的 `policy.onnx` 与
运行时所用的部署侧配置和运动资产放在一起。

## G1 部署原型

已提交的 G1 WBT 部署辅助工具使用如下产物：

| 产物 | 生产者 |
| --- | --- |
| `policy.onnx` | 上述训练回放导出。 |
| `deploy_config.yaml` | `scripts/deploy/export_deploy_config.py`。 |
| `dance1.bin` 或其他运动二进制 | `scripts/deploy/export_motion_bin.py`。 |

验证运行示例：

```bash
uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/export_motion_bin.py \
  --output logs/deploy/dance1.bin

uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

`scripts/deploy/sim_prototype.py` 会检查 ONNX 输入宽度是否与 `deploy_config.yaml` 中的
`obs_dim` 匹配，然后用部署侧期望的同一观测布局在 MuJoCo 中驱动策略。

## 另请参阅

- {doc}`8-latency_budget`
- {doc}`7-safety_layers`
