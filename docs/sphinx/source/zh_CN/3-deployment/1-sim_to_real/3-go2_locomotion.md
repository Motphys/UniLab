# Go2 / Go2W 运动部署

语言: 简体中文

摇杆驱动的运动（平地 + 崎岖）以及轮足式的 Go2W 变体。两者的硬件流程相似；本页指出
其中的差异。

## 观测契约

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - 分组
  - 维度
  - 硬件上的来源
* - 基座线速度
  - 3
  - 状态估计器（在 IMU + 腿部里程计上的 KF）；不是原始积分
* - 基座角速度
  - 3
  - IMU 陀螺仪
* - 投影重力
  - 3
  - IMU 朝向
* - 摇杆指令 (vx, vy, ωz)
  - 3
  - 操作员输入
* - 关节位置
  - 12（Go2）/ 16（Go2W）
  - 编码器
* - 关节速度
  - 12 / 16
  - 经过部署控制器滤波路径后的编码器速度
* - 上一步动作
  - 12 / 16
  - 上一次策略输出
* - 足端接触
  - 4（仅 Go2）
  - 接触传感器，或由足端高度估计
```

::::{admonition} 状态估计器注意事项
:class: warning
策略是针对所选环境 owner 发出的观测项训练的。如果部署无法提供同样的基座速度信号，
请训练一个变体，使其 actor 观测与你能在机器人上运行的估计器相匹配（参见 HIM-PPO，
见 {doc}`../../2-user_guide/2-algorithms/6-him_ppo`）。
::::

## 崎岖地形注意事项

对于 `go2_joystick_rough`，策略期望存在抬升的地形特征。在平坦的室内地面上，按崎岖
地形训练的策略会比必要时*更加保守*，但在硬件上机前仍应通过回放进行验证。对于在
斜坡 / 碎屑上的部署：

- 从实测的部署表面选取地面摩擦的 DR 范围。
- 用地形课程训练：见
  {doc}`../../2-user_guide/6-terrain/1-procedural`。

## Go2W 轮 ↔ 腿分派

Go2W 策略为后轮关节输出**连续轮速**，并为腿部输出**位置目标**。动作向量的顺序必须
与 `src/unilab/assets/robots/go2w/` 匹配。用 `unilab-export-scene` 验证。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/4-tasks/1-locomotion`

## Navigation

- Index: [文档](0-index.md)
