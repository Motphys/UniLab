# 延迟预算

本页记录仓库中可见的延迟控制项，以及硬件上机前你需要做的部署侧测量。把数值预算当作
机器人专属的测量结果，而不是 UniLab 的默认值。

## 仓库中的延迟面

| 面 | 仓库证据 | 它覆盖什么 |
| --- | --- | --- |
| 单步动作延迟 | locomotion 与 G1 运动跟踪环境中的 `control_config.simulate_action_latency` | 执行上一步动作而非当前动作。 |
| G1 WBT 观测历史 | `noise_config.obs_history_length` 与 `scripts/deploy/export_deploy_config.py` | 为 `gyro`、`joint_pos_rel`、`dof_vel` 与 `last_actions` 导出逐项的 `obs_layout` 历史。 |
| Sharpa 触觉接触延迟 | Sharpa 手内配置中的 `domain_rand.contact_latency` | 为采样到的接触通道保留上一步的触觉接触值。 |
| 部署侧 ONNX 契约检查 | `scripts/deploy/sim_prototype.py` | 为 G1 WBT 路径校验 `obs_layout`、`obs_dim`、ONNX 输入宽度、钳制以及 EMA 动作平滑。 |

## 动作延迟

对于暴露 `control_config.simulate_action_latency` 的任务，当该开关启用时，环境会应用
`last_actions`。把它保留在所选的任务 owner YAML 中，而不要事后添加仅部署的行为。

```yaml
env:
  control_config:
    simulate_action_latency: true
```

已签入的 G1 WBT owner 在 `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` 中启用了
该开关。

## 观测滞后与历史

G1 WBT 部署辅助工具不会猜测观测宽度。它们导出一份带有 `obs_layout`、逐项
`history_length` 与 `obs_dim` 的模式；随后原型组装出同样的布局，并拒绝不匹配。

```bash
uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

除非训练 owner 这样做了，否则不要让指令/参考项滞后。在 G1 WBT 模式中，参考项保持
单步，而本体感受项携带历史。

## 部署侧测量

在硬件运行时，对每个策略 tick 记录如下内容：

1. `policy_input_timestamp`
2. 每个传感器或估计器通道的源时间戳
3. `policy_output_timestamp`
4. 执行器指令的发送时间戳
5. 钳制 / 平滑前后的动作向量

将观测向量与用同一份 `deploy_config.yaml` 构建的仿真回合作对比。如果实测管线需要
滤波或缓冲，请把相匹配的行为编码到任务 owner 中，并重新导出部署产物。

## 不匹配的症状

- 使能力矩后出现接触振荡。
- 在最初几个策略 tick 期间出现动作饱和。
- 即便 ONNX 输入宽度与观测布局匹配，仍出现速度跟踪漂移。

## 另请参阅

- {doc}`6-domain_randomization`
- {doc}`7-safety_layers`
- `src/unilab/dr/manager.py`
