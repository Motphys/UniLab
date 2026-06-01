# 工具

用于导出策略、检查训练失败、发送运行元数据以及实例化场景的运维工具。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} ONNX 导出
:link: 1-onnx_export
:link-type: doc
从回放路径导出策略并验证运行时输入。
:::

:::{grid-item-card} W&B 与 TensorBoard
:link: 2-wandb
:link-type: doc
配置运行日志与实验元数据。
:::

:::{grid-item-card} NaN 可视化工具
:link: 3-nan_visualizer
:link-type: doc
检查 PPO 运行中的 NaN guard dump。
:::

:::{grid-item-card} 场景导出
:link: 4-scene_export
:link-type: doc
导出 MuJoCo 场景及复制的 asset 以供检查。
:::

::::

```{toctree}
:hidden:

1-onnx_export
2-wandb
3-nan_visualizer
4-scene_export
```
