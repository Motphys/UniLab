# 延迟预算

本页记录仓库中可见的延迟控制项，以及硬件上机前你需要做的部署侧测量。把数值预算当作
机器人专属的测量结果，而不是 UniLab 的默认值。

## 仓库中的延迟面

| 面 | 仓库证据 | 它覆盖什么 |
| --- | --- | --- |
| 单步动作延迟 | task owner 中 Manager action term 的 `simulate_action_latency` 声明 | 执行上一步动作而非当前动作。 |
| G1 WBT 观测历史 | `src/unilab/conf/sac/task/g1_wbt_obs/mujoco.yaml` 中逐 term 的 `history_length` | 为 `base_ang_vel`、`joint_pos`、`joint_vel` 与 `actions` 提供逐项历史。 |
| Sharpa 触觉接触延迟 | Sharpa 手内配置中的 `domain_rand.contact_latency` | 为采样到的接触通道保留上一步的触觉接触值。 |
| 观测历史顺序守护 | `tests/scripts/test_obs_alignment_g1_wbt.py` | 断言 G1 WBT actor 观测按逐项最旧优先展平。 |

## 动作延迟

对于启用 action latency 的 Manager-Based 任务，action manager 会在该开关开启时应用
上一步 action。把它保留在所选的 task owner YAML 中，而不要事后添加仅部署的行为。

```yaml
env:
  actions:
    joint_pos:
      simulate_action_latency: true
```

已签入的 G1 WBT owner 在 `src/unilab/conf/sac/task/g1_wbt_obs/mujoco.yaml` 中启用了
该开关。

## 观测滞后与历史

观测宽度是所声明 actor 各项 `dim * history_length` 之和，硬件运行时不允许猜测。对
G1 WBT owner，`history_length: 5` 让每个本体感受项携带 5 步历史并按最旧优先展平，
而参考项保持单步。完整的分项顺序见 {doc}`2-g1_whole_body`。

除非训练 owner 这样做了，否则不要让指令/参考项滞后。

## 部署侧测量

在硬件运行时，对每个策略 tick 记录如下内容：

1. `policy_input_timestamp`
2. 每个传感器或估计器通道的源时间戳
3. `policy_output_timestamp`
4. 执行器指令的发送时间戳
5. 钳制 / 平滑前后的动作向量

将观测向量与用同一份任务 owner YAML 构建的仿真回合作对比。如果实测管线需要滤波或
缓冲，请把相匹配的行为编码到任务 owner 中并重新训练，而不是只在部署侧添加。

## 不匹配的症状

- 使能力矩后出现接触振荡。
- 在最初几个策略 tick 期间出现动作饱和。
- 即便 ONNX 输入宽度与观测布局匹配，仍出现速度跟踪漂移。

## 另请参阅

- {doc}`6-domain_randomization`
- {doc}`7-safety_layers`
- `src/unilab/dr/manager.py`
