# ONNX 运行时

UniLab 从既有的训练回放路径导出 ONNX 策略。使用产出该检查点的同一算法家族与任务
owner；回放代码加载检查点、导出 `policy.onnx`，并在该路径实现了 ONNX Runtime 检查时
校验所导出的计算图。

## 导出路径

| 算法路径 | 入口脚本 | 仓库中的导出行为 |
| --- | --- | --- |
| PPO（torch） | `src/unilab/scripts/train_rsl_rl.py` | 脚本入口处 `EXPORT_POLICY=True`；回放调用 `runner.export_policy_to_onnx(...)` 与 `runner.export_policy_to_jit(...)`。 |
| HIM-PPO | `scripts/train_him_ppo.py` | 与 PPO 相同的脚本级导出模式。 |
| APPO | `src/unilab/scripts/train_appo.py` | 回放写出 `policy.onnx` 并将 ONNX Runtime 输出与 PyTorch 比对校验。 |
| SAC / TD3 / FlashSAC | `src/unilab/scripts/train_sac.py` / `src/unilab/scripts/train_td3.py` / `src/unilab/scripts/train_flashsac.py` | 回放写出 `policy.onnx`；SAC 与 FlashSAC 在导出前使用 `actor.as_export_module()`。 |

## 命令

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo appo --task g1_motion_tracking --sim motrix --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` 设置回放模式，并把 `--load-run` 映射到所路由训练脚本使用的检查点
选择器。导出的文件会写入所选的运行目录。用于部署时，请把导出的 `policy.onnx` 与训练
它所用的任务 owner YAML 放在一起——该 YAML 是运行时必须复现的观测与动作契约的权威
来源。

## 校验导出的计算图

回放路径在写出之前会把导出的计算图与 PyTorch 比对，因此导出成功本身已经建立了数值
一致性。它**没有**建立的是：你的硬件侧回路装配出的输入向量是否相同。在硬件上机前：

- 从 composed config 读取 actor 观测宽度（而不是照抄文档表格），确认它与 ONNX
  输入宽度一致。
- 对照 owner 的 `env.observations.actor.terms` 确认分项顺序与逐项历史顺序。G1
  全身跟踪见 {doc}`2-g1_whole_body`。

## 另请参阅

- {doc}`8-latency_budget`
- {doc}`7-safety_layers`
