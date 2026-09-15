# Go2 运动部署

本页描述保留的 `go2_joystick_flat` reference owner。Unitree production 部署
变体由对应 ecosystem 包文档维护。

## 观测契约

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Group
  - Dim
  - 硬件来源
* - Base linear velocity
  - 3
  - 状态估计器（IMU + 足端里程计 KF）；不要使用原始积分
* - Base angular velocity
  - 3
  - IMU 陀螺仪
* - Projected gravity
  - 3
  - IMU 姿态
* - Joystick command (vx, vy, ωz)
  - 3
  - 操作者输入
* - Joint positions
  - 12
  - 编码器
* - Joint velocities
  - 12
  - 部署控制器滤波后的编码器速度
* - Previous action
  - 12
  - 上一步策略输出
* - Foot contact
  - 4
  - 接触传感器或由足端高度估计
```

::::{admonition} 状态估计器注意事项
:class: warning
策略训练时使用所选 env owner 声明的观测 term。如果部署侧无法提供相同的
base-velocity 信号，请训练 actor 观测与机器人可用估计器匹配的变体。
::::

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/4-tasks/1-locomotion`
