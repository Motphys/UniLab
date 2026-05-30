# 快速上手

如果你正从一个全新的检出（checkout）配置 UniLab，或者想用最短的路径完成第一次冒烟运行，请从这里开始。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 快速演示
:link: 1-quick_demo
:link-type: doc
克隆仓库、同步 Motrix 依赖、运行 PPO，然后进行评估或演示。
:::

:::{grid-item-card} 安装
:link: 2-installation
:link-type: doc
配置 `uv`、同步依赖，并为你的机器选择对应的平台配置档（profile）。
:::

:::{grid-item-card} 评估与回放
:link: 3-evaluation_and_playback
:link-type: doc
回放检查点（checkpoint）、导出视频，以及使用演示模式。
:::

:::{grid-item-card} 项目结构
:link: 4-project_structure
:link-type: doc
找到负责 scripts、configs、envs、backends 和 docs 的各个目录。
:::

::::

完成第一次运行后，日常训练选项请参阅
{doc}`../2-user_guide/1-training/0-index`。

```{toctree}
:hidden:
:caption: Getting Started

1-quick_demo
2-installation
3-evaluation_and_playback
4-project_structure
```
