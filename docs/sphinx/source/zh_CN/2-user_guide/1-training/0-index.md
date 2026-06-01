# 训练

UniLab 中的训练以配置为先（config-first）。日常运行使用软件包 CLI，调试底层
Hydra 组合时再使用脚本入口。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} CLI 参考
:link: 1-cli_reference
:link-type: doc
`uv run train`、`uv run eval`、`uv run demo` 以及底层脚本的路由。
:::

:::{grid-item-card} Hydra 配置
:link: 2-hydra_config
:link-type: doc
owner YAML 布局、后端选择以及安全的 override 示例。
:::

:::{grid-item-card} 日志与跟踪
:link: 3-logging
:link-type: doc
TensorBoard、W&B、运行元数据以及 trace 选项。
:::

:::{grid-item-card} 续训与检查点
:link: 5-resume_and_checkpoints
:link-type: doc
`algo.load_run`、检查点文件与回放命令之间如何配合。
:::

:::{grid-item-card} Docker
:link: 6-docker
:link-type: doc
在仓库内置的 Linux NVIDIA 镜像工作流中运行 UniLab。
:::

:::{grid-item-card} 多 GPU
:link: 4-multi_gpu
:link-type: doc
当前 off-policy 的多 GPU 旋钮及其配置边界。
:::

::::

## 什么时候下沉到 `scripts/train_*.py`

日常运行应使用统一 CLI。只有在下面几种场景才直接使用底层
`scripts/train_*.py` 入口：

- 需要调试特定训练栈，
- 需要直接观察 Hydra compose 行为，或
- 需要对照脚本级日志目录或 adapter 行为。

## 关联

- 选择算法：{doc}`../2-algorithms/0-index`
- 查找任务命令：{doc}`../4-tasks/0-index`
- 对比后端行为：{doc}`../3-backends/0-index`
- 查询精确支持状态：{doc}`../../5-reference/5-support_matrix`

```{toctree}
:hidden:

1-cli_reference
2-hydra_config
3-logging
4-multi_gpu
5-resume_and_checkpoints
6-docker
```
