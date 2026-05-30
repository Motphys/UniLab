# 仿真到真机

为硬件上机准备一个训练好的 UniLab 策略。先从总览开始，然后依次走完导出、随机化、
安全、延迟以及机器人专属的说明。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 总览与上机前检查
:link: 1-overview
:link-type: doc
端到端流程与 go/no-go 清单。
:::

:::{grid-item-card} ONNX 运行时
:link: 5-onnx_runtime
:link-type: doc
训练回放导出、ONNX Runtime 检查以及部署输入。
:::

:::{grid-item-card} 域随机化
:link: 6-domain_randomization
:link-type: doc
面向真机迁移、按优先级排序的随机化检查。
:::

:::{grid-item-card} 安全层
:link: 7-safety_layers
:link-type: doc
软限位、动作滤波、看门狗与急停边界。
:::

:::{grid-item-card} 延迟预算
:link: 8-latency_budget
:link-type: doc
训练侧的延迟旋钮与部署侧的测量检查。
:::

:::{grid-item-card} 故障排查
:link: 9-troubleshooting
:link-type: doc
面向硬件上机的症状、原因与修复说明。
:::

:::{grid-item-card} G1 全身
:link: 2-g1_whole_body
:link-type: doc
G1 路线的运动跟踪部署说明。
:::

:::{grid-item-card} Go2 运动
:link: 3-go2_locomotion
:link-type: doc
摇杆、崎岖地形与 Go2W 的部署说明。
:::

:::{grid-item-card} Allegro 手内操作
:link: 4-allegro_inhand
:link-type: doc
手内操作部署检查。
:::

::::

```{toctree}
:hidden:

1-overview
2-g1_whole_body
3-go2_locomotion
4-allegro_inhand
5-onnx_runtime
6-domain_randomization
7-safety_layers
8-latency_budget
9-troubleshooting
```
