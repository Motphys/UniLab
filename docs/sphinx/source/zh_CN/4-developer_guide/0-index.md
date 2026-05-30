# Developer 指南

当你要修改 UniLab 本身时使用本章节：运行时契约、后端能力、任务 owner、
算法、测试或贡献流程。

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 架构概览
:link: 1-architecture/1-overview
:link-type: doc
运行时模型、分层所有权、config-first 规则与验证标准。
:::

:::{grid-item-card} Registry
:link: 1-architecture/5-registry
:link-type: doc
Bootstrap 导入、env 注册与运行时构造。
:::

:::{grid-item-card} Env 契约
:link: 2-contracts/1-env_contract
:link-type: doc
`NpEnvState`、reset/step 形状、observation 分组与 wrapper 预期。
:::

:::{grid-item-card} Backend 契约
:link: 2-contracts/2-backend_contract
:link-type: doc
`SimBackend` 边界与可选能力（capability）模式。
:::

:::{grid-item-card} 任务 owner 契约
:link: 2-contracts/3-task_owner
:link-type: doc
Hydra owner YAML 身份与后端选择规则。
:::

:::{grid-item-card} 域随机化契约
:link: 2-contracts/4-dr_contract
:link-type: doc
DR provider 的 init、reset、interval 与后端能力边界。
:::

::::

## 扩展

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 新任务
:link: 3-extending/1-new_task
:link-type: doc
添加 env config、注册、owner YAML 与测试。
:::

:::{grid-item-card} 新后端
:link: 3-extending/2-new_backend
:link-type: doc
添加 `SimBackend` 实现并显式声明能力支持。
:::

:::{grid-item-card} 新算法
:link: 3-extending/3-new_algorithm
:link-type: doc
添加配置、runner 代码与脚本层组装，且不改动 env 契约。
:::

:::{grid-item-card} 新地形
:link: 3-extending/4-new_terrain
:link-type: doc
扩展地形生成，同时将资源访问保持在冷路径上。
:::

::::

## 贡献者工作流

- {doc}`4-contributing`
- {doc}`5-contributing_workflow`
- {doc}`6-agent_quick_reference`
- {doc}`ADR index </adr/ADR-0000-index>`

```{toctree}
:hidden:
:caption: Developer Guide

1-architecture/0-index
2-contracts/0-index
3-extending/0-index
4-contributing
5-contributing_workflow
6-agent_quick_reference
```
