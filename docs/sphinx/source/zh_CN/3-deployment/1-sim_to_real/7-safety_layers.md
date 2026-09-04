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

把硬实时的安全检查放在部署控制器中，而不是训练脚本里。仓库不实现生产级的电机驱动器
安全回路——该边界由你构建并测试。

## 策略假定你已配置的内容

策略期望的是其训练 owner 所声明的动作映射与限位。对 G1 WBT owner
（`src/unilab/conf/sac/task/g1_wbt_obs/mujoco.yaml`）：

| 量 | 权威来源 |
| --- | --- |
| 动作缩放 | `env.actions.joint_pos.scale`（该 owner 为标量 `2.0`；其他 owner 声明按 actuator 解析的正则 → 数值映射） |
| 默认关节角 | `use_default_offset: true`，即 owner 场景 XML 中 `stand` keyframe 的关节段 |
| 关节限位 | 场景 XML 中的 `jnt_range` |
| `kp` / `kd` | 场景 XML 中 position actuator 的 `gainprm` / `biasprm` |

请从 owner YAML 及其场景派生这些量，并原样复现 owner 解析后的逐 actuator 缩放——
标量 owner 与正则映射型 owner 不可互换。不要把关节范围或增益手工复制到第二处，那会
与资产静默漂移。

## 交接测试

在把 策略 → 安全层 → 电机 集成起来之前，先隔离测试安全层：

1. 注入一个 NaN 动作，验证该指令被拒绝。
2. 注入一个超范围的关节目标，验证钳制使用了 owner 场景 XML 中的关节范围。
3. 在运行途中切断策略输入，验证控制器进入其配置的安全状态。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`9-troubleshooting`
- {doc}`2-g1_whole_body`
