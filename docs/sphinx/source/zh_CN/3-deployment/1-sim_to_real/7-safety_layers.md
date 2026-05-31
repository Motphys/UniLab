# 硬件安全层

策略在训练契约下产生动作。一个部署侧的安全层必须位于**策略输出与电机驱动器之间**，
并在契约违例成为执行器指令之前将其拒绝。

## 必备组件

```{list-table}
:header-rows: 1
:widths: 30 70

* - 层
  - 职责
* - 模式检查
  - 动作具有正确的 dtype、形状、有限值。拒绝 NaN / Inf。
* - 范围钳制
  - 将每个关节目标钳制到部署配置的关节限位。
* - Δ 钳制
  - 使用部署控制器拥有的阈值，拒绝或钳制逐步的动作增量。
* - 速率限制
  - 在钳制之后施加变化率限制。
* - 看门狗
  - 若在控制器拥有的超时内没有新的动作到达，则保持最后一个已知的安全目标，或进入
    控制器的安全状态。
* - 姿态监控
  - 横滚 / 俯仰超出工作包络 → 触发故障。
* - 操作员停止
  - 大红按钮 → 立即关闭力矩，无论处于何种状态。
```

## 安全层位于何处

```{mermaid}
flowchart LR
    P[Policy ONNX] --> S[Safety layer<br/>C++ on robot computer]
    S -->|safe target| D[Motor driver]
    D -->|encoder + IMU| Pre[Observation builder]
    Pre --> P
    S -.->|fault| OP[Operator UI]
    OP -.->|E-stop| D
```

把硬实时的安全检查放在部署控制器中，而不是训练脚本里。仓库的 G1 辅助路径导出部署
配置并运行一个 MuJoCo 原型；它并不实现生产级的电机驱动器安全回路。

## 策略假定你已配置的内容

G1 部署辅助工具会把这些字段导出到 `deploy_config.yaml`：

```yaml
action_scale: 2.0
ema_alpha: 1.0
default_angles: [...]
joint_lower: [...]
joint_upper: [...]
kp: [...]
kd: [...]
```

`scripts/deploy/sim_prototype.py` 消费同样的字段，并应用
`action * action_scale + default_angles`、关节钳制与 EMA 平滑。硬件控制器应当消费
生成的配置，而不是手动复制关节范围或增益。

## 交接测试

在把 策略 → 安全层 → 电机 集成起来之前，先隔离测试安全层：

1. 注入一个 NaN 动作，验证该指令被拒绝。
2. 注入一个超范围的关节目标，验证钳制使用了 `deploy_config.yaml` 中的
   `joint_lower` / `joint_upper`。
3. 在运行途中切断策略输入，验证控制器进入其配置的安全状态。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`9-troubleshooting`
- {doc}`2-g1_whole_body`
