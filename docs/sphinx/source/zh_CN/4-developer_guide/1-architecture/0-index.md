# 架构

架构相关页面概述了运行时模型、所有权边界、场景组合规则以及 registry
bootstrap 契约。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 概览
:link: 1-overview
:link-type: doc
运行时模型、分层所有权与验证标准。
:::

:::{grid-item-card} 运行时模型
:link: 2-runtime_model
:link-type: doc
Runner 生命周期、worker/learner 拆分与数据流。
:::

:::{grid-item-card} 分层边界
:link: 3-layer_boundaries
:link-type: doc
针对 scripts、envs、backends 与算法的 owner 层规则。
:::

:::{grid-item-card} 场景组合
:link: 4-scene_composition
:link-type: doc
场景 fragment、资源与冷路径 materialization。
:::

:::{grid-item-card} Registry
:link: 5-registry
:link-type: doc
Bootstrap 导入与 env/backend 注册。
:::

::::

```{toctree}
:hidden:

1-overview
2-runtime_model
3-layer_boundaries
4-scene_composition
5-registry
```
