---
sd_hide_title: true
---

# UniLab 文档

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab

### 一次配置任务语义，跨物理后端运行机器人 RL。

{bdg-primary}`Python >=3.10,<3.14` {bdg-secondary}`Hydra + Manager API` {bdg-info}`跨后端 contract` {bdg-success}`uv workflow`

UniLab 将任务语义变成可复用配置：组装 manager term、选择物理后端，并在手头的硬件
上运行同一套训练/评估工作流。可以从这个着陆页开始：安装、运行第一次 demo、再做
冒烟训练、选择算法/后端，或直接跳到部署与扩展文档。

```{button-ref} 1-getting_started/1-quick_demo
:ref-type: doc
:color: primary
:class: sd-px-4 sd-py-2

快速演示
```
```{button-ref} 2-user_guide/0-index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4 sd-py-2

用户指南
```
:::

::::

## 为什么选择 UniLab

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 配置任务，无需样板代码
在 Hydra owner YAML 中用 manager term 组装 action、observation、reward、termination、
event、command 和 curriculum。常见任务变体无需新写 environment class。
:::

:::{grid-item-card} 后端选择留在配置里
用 CLI flag 在当前和未来的物理 adapter 之间切换，例如
`--task go2_joystick_flat --sim motrix`；CLI 会组合 `src/unilab/conf/` 下对应的 owner YAML。
:::

:::{grid-item-card} 跨硬件扩展
任务 contract 将 CPU 并行或外部 worker 仿真连接到 accelerator learner，实验可以随着
手头可用的硬件持续扩展。
:::

::::

## 快速安装与冒烟运行

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
uv run demo dance
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true
```

完整的 README 风格演练见 {doc}`1-getting_started/1-quick_demo`。
平台相关的安装见 {doc}`1-getting_started/2-installation`。

## 从你所处的位置开始

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 安装仓库
:link: 1-getting_started/2-installation
:link-type: doc
配置 `uv`、同步依赖，并选择与你机器匹配的平台 profile。
:::

:::{grid-item-card} 运行或回放训练
:link: 1-getting_started/1-quick_demo
:link-type: doc
先运行预训练 demo，再进入 PPO 训练、评估、回放或 checkpoint 续训。
:::

:::{grid-item-card} 选择物理后端
:link: 2-user_guide/3-backends/0-index
:link-type: doc
通过 task owner YAML 选择后端，并阅读其安装与能力要求。
:::

:::{grid-item-card} 挑选算法
:link: 2-user_guide/2-algorithms/0-index
:link-type: doc
对比 PPO、APPO、SAC、TD3、FlashSAC、HIM-PPO 和 HORA 的入口。
:::

:::{grid-item-card} 部署或切换仿真
:link: 3-deployment/1-sim_to_real/1-overview
:link-type: doc
按 sim-to-real 检查清单操作，或用 sim-to-sim 文档在 MuJoCo 与
Motrix 之间互换。
:::

:::{grid-item-card} 安全地扩展
:link: 4-developer_guide/0-index
:link-type: doc
在新增任务、后端、算法或地形之前，先阅读 env、backend、runner、
registry 和 task-owner contract。
:::

::::

## 架构速览

```{mermaid}
flowchart LR
  cli["uv run train/eval<br/>--algo --task --sim"] --> owner["Task owner YAML<br/>src/unilab/conf/*/task/..."]
  cli --> script["Thin script routing<br/>src/unilab/scripts/train_*.py"]
  owner --> registry["Registry bootstrap<br/>src/unilab/base/registry.py"]
  registry --> env["NpEnv contract<br/>obs dict + info dict"]
  env --> backend["SimBackend<br/>unisim-core adapters"]
  env --> factory["EnvFactory contract"]
  factory --> runtime["Runner / IPC<br/>unilab-rl async runtime"]
  runtime --> learner["Learner<br/>PPO / APPO / SAC / TD3"]
```

承载核心的 contract 记录在
{doc}`4-developer_guide/0-index`；后端支持证据汇总于
{doc}`2-user_guide/3-backends/0-index`。

## 硬件与算法覆盖

这份速览只列出有已提交脚本、owner YAML 和所生成支持矩阵证据等级支撑的
覆盖情况。仓库目前没有已提交的 benchmark manifest，也没有单独的
recommendation 元数据。

| 机器人 / 任务族 | 有仓库证据的算法路径 | 后端证据 |
| --- | --- | --- |
| Go1 joystick | PPO、APPO、TD3 | PPO 有已测试的 MuJoCo 与 Motrix 行。APPO 有已测试的 MuJoCo 行和 Motrix registered 行。TD3 有 `go1_joystick_flat` 的 Motrix owner YAML。 |
| Go2 joystick | PPO、FlashSAC、TD3 | PPO 有已测试的 MuJoCo 与 Motrix 行。FlashSAC 有 `go2_joystick_flat` 的 MuJoCo owner YAML；TD3 有 `go2_joystick_flat` 的 Motrix owner YAML。 |
| Go2 arm manip-loco | PPO、HIM-PPO | `src/unilab/conf/ppo/task/go2_arm_manip_loco/` 和 `src/unilab/conf/ppo_him/task/go2_arm_manip_loco/` 下有已提交的 MuJoCo owner YAML。 |
| Go2W joystick | PPO | `src/unilab/conf/ppo/task/go2w_joystick_*` 下存在 MuJoCo 与 Motrix flat/rough 变体的 PPO owner YAML。 |
| G1 locomotion / tracking | PPO、APPO、SAC、TD3 | PPO、APPO、SAC 都为 G1 任务提供了已提交的 MuJoCo 与 Motrix owner YAML；TD3 有一个 `g1_walk_flat` 的 MuJoCo owner。 |
| Allegro in-hand | PPO、APPO | PPO 和 APPO 为 Allegro in-hand 任务提供了已提交的 MuJoCo 与 Motrix owner YAML。 |
| Sharpa in-hand | PPO、APPO HORA teacher、HORA distillation | Sharpa owner YAML 为 PPO/APPO teacher 路径已提交；student distillation 使用 `src/unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml`。 |

```{toctree}
:hidden:
:caption: 文档

1-getting_started/0-index
2-user_guide/0-index
3-deployment/0-index
4-developer_guide/0-index
5-reference/0-index
```
