# 仿真到真机总览

本页是 UniLab 仿真到真机工作流的*地图*。本节后续的每一页都会深入其中一个阶段。

## "仿真到真机"在 UniLab 中的含义

一个可部署的 UniLab 策略，是导出的策略加上所选任务 owner 使用的那套精确的观测与
动作契约。G1 WBT 辅助路径将其物化为 `policy.onnx`、`deploy_config.yaml` 以及一个
运动二进制文件；其他机器人需要一个等价的硬件侧运行时，它需要：

1. 读取传感器 → 组装出策略在仿真中看到的**同一个观测向量**。
2. 通过一个支持所导出计算图的运行时来运行 `policy.onnx`。
3. 将动作向量映射到环境 `SimBackend` 所使用的同一套执行器接口。

如果这三件事中的任何一件在仿真与部署之间发生漂移，请先调试契约，再去改奖励或硬件
调参。

## 端到端流程

```{mermaid}
flowchart LR
    A[Train in UniLab] --> B[Curriculum + DR]
    B --> C[Validate in alt backend]
    C --> D[Export ONNX]
    D --> E[Latency / lag injection]
    E --> F[Safety layer]
    F --> G[Hardware bringup]
    G --> H[Closed-loop run]
    H -. iterate .-> B
```

| 阶段 | UniLab 产物 | 页面 |
|---|---|---|
| 训练 | 任务 owner YAML + 训练脚本 | {doc}`../../2-user_guide/1-training/1-cli_reference` |
| 课程 + DR | `unilab.dr` + 任务侧 provider | {doc}`6-domain_randomization` |
| 跨后端健全性检查 | `--task <task> --sim <other_backend>` | {doc}`../2-sim_to_sim/1-backend_swap` |
| ONNX 导出 | 训练回放脚本 + 部署辅助工具 | {doc}`5-onnx_runtime` |
| 延迟 / 观测滞后 | 任务配置开关与部署侧日志 | {doc}`8-latency_budget` |
| 安全层 | 硬件侧钳制 / 回退 | {doc}`7-safety_layers` |
| 机器人上机 | 机器人专属指南 | {doc}`2-g1_whole_body`、{doc}`3-go2_locomotion`、{doc}`4-allegro_inhand` |

## 开始之前你应当具备的条件

::::{admonition} 上机前检查清单
:class: note

1. 一次**收敛的训练运行**，奖励稳定，且成功判据也稳定（运动跟踪误差、跌落次数等）。
2. 当 MuJoCo 与 Motrix **都**支持该任务时，同一策略在两者中都能通过评估 —— 否则
   你存在一个依赖后端的奖励泄漏；见 {doc}`../2-sim_to_sim/4-reward_parity`。
3. **域随机化**范围足够大，使得在扫动 DR 强度时奖励平滑变化 —— 在仿真里脆弱的策略
   在硬件上同样脆弱。
4. 环境中**没有后端功能泄漏** —— 通过开发者指南的
   {doc}`../../4-developer_guide/2-contracts/2-backend_contract` 验证。
5. 一份你能在硬件上实现的**观测规格**。如果你的策略读取 `body_lin_vel`，你需要一个
   部署侧的估计器，或者一个把该信号从 actor 输入中移除的任务 owner 变体。

::::

## 最常见的失败模式

- **观测漂移。** 仿真与部署运行时之间的传感器预处理不同（单位、坐标系、滤波截止频率）。
  记录第一段部署侧观测窗口，并与用同一份 owner YAML 构建的仿真回合作对比。
- **动作延迟。** 一些任务配置通过 `control_config.simulate_action_latency` 暴露单步
  延迟的动作执行。测量部署回路，并在硬件运行前让训练 owner 匹配该契约。见
  {doc}`8-latency_budget`。
- **摩擦 / 阻尼不匹配。** 尤其对于手内操作。在 DR 中扫动摩擦；通过
  {doc}`../2-sim_to_sim/3-contact_and_friction_alignment` 交叉核对。
- **复位瞬态。** 仿真复位到一个稳定姿态；部署则从一个控制器状态开始。安全层必须在
  畸形观测与不安全动作到达电机驱动器之前将其拒绝。

## 各机器人快速链接

::::{grid} 3
:gutter: 2

:::{grid-item-card} 🤖 G1 全身
:link: 2-g1_whole_body
:link-type: doc

人形运动跟踪部署、关节钳制范围、IMU 对齐。
:::

:::{grid-item-card} 🐕 Go2 运动
:link: 3-go2_locomotion
:link-type: doc

Go2 与 Go2W 上的摇杆 + 崎岖地形策略。
:::

:::{grid-item-card} ✋ Allegro 手内操作
:link: 4-allegro_inhand
:link-type: doc

灵巧的方块重定向、无触觉部署、抓取生成器。
:::

::::
