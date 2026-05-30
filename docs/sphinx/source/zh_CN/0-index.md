---
sd_hide_title: true
---

# UniLab 文档

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab

### 面向 CPU 仿真与加速器训练的 contract 驱动机器人学习基础设施。

{bdg-primary}`Python >=3.10,<3.14` {bdg-secondary}`Hydra owner YAML` {bdg-info}`MuJoCo + Motrix` {bdg-success}`uv workflow`

UniLab 通过 `uv run train` / `uv run eval` CLI、task-owner Hydra 配置和 backend
contract 来组织机器人 RL。可以从这个着陆页开始：安装、跑一次冒烟训练、选择
算法/后端，或直接跳到部署与扩展文档。

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

:::{grid-item-card} CPU 仿真，加速器训练
README 把 UniLab 描述为通过共享内存把 CPU 物理仿真连接到策略训练，
以 MuJoCo 和 Motrix 作为仿真后端。
:::

:::{grid-item-card} 后端选择留在配置里
用 CLI flag 切换后端，例如 `--task go2_joystick_flat --sim motrix`；
CLI 会组合 `conf/` 下对应的 owner YAML。不要把
`training.sim_backend` 当作独立的后端开关。
:::

:::{grid-item-card} 部署路径都有文档
部署文档覆盖 sim-to-real、sim-to-sim、ONNX/runtime 导出、安全
层，以及 G1、Go2、Allegro 的机器人专属说明。
:::

::::

## 快速安装与冒烟运行

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
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
先在 Go2 上跑 PPO，再进入评估、回放或 checkpoint 续训。
:::

:::{grid-item-card} 选择后端
:link: 2-user_guide/3-backends/3-choosing_a_backend
:link-type: doc
通过 task owner YAML 和后端能力文档对比 MuJoCo 与 Motrix。
:::

:::{grid-item-card} 挑选算法
:link: 2-user_guide/2-algorithms/0-index
:link-type: doc
对比 PPO、APPO、SAC、TD3、FlashSAC、MLX PPO、HIM-PPO 和 HORA 的入口。
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
  cli["uv run train/eval<br/>--algo --task --sim"] --> owner["Task owner YAML<br/>conf/*/task/..."]
  cli --> script["Thin script routing<br/>scripts/train_*.py"]
  owner --> registry["Registry bootstrap<br/>src/unilab/base/registry.py"]
  registry --> env["NpEnv contract<br/>obs dict + info dict"]
  env --> backend["SimBackend<br/>MuJoCo or Motrix"]
  env --> runtime["Runner / IPC<br/>shared memory lifecycle"]
  runtime --> learner["Learner<br/>PPO / APPO / SAC / TD3 / MLX"]
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
| Go1 joystick | PPO (torch, MLX)、APPO、TD3 | PPO 有已测试的 MuJoCo 与 Motrix 行。APPO 有已测试的 MuJoCo 行和 Motrix registered 行。TD3 有 `go1_joystick_flat` 的 Motrix owner YAML。 |
| Go2 joystick / handstand | PPO (torch, MLX)、FlashSAC、TD3 | PPO 有已测试的 MuJoCo 与 Motrix 行。FlashSAC 有 `go2_joystick_flat` 的 MuJoCo owner YAML；TD3 有 `go2_joystick_flat` 的 Motrix owner YAML。 |
| Go2 arm manip-loco | PPO、HIM-PPO | `conf/ppo/task/go2_arm_manip_loco/` 和 `conf/ppo_him/task/go2_arm_manip_loco/` 下有已提交的 MuJoCo owner YAML。 |
| Go2W joystick | PPO (torch、MLX configured) | `conf/ppo/task/go2w_joystick_*` 下存在 MuJoCo 与 Motrix flat/rough 变体的 PPO owner YAML。 |
| G1 locomotion / tracking | PPO (torch, MLX)、APPO、SAC、TD3 | PPO、APPO、SAC 都为 G1 任务提供了已提交的 MuJoCo 与 Motrix owner YAML；TD3 有一个 `g1_walk_flat` 的 MuJoCo owner。 |
| Allegro in-hand | PPO (torch、MLX configured)、APPO | PPO 和 APPO 为 Allegro in-hand 任务提供了已提交的 MuJoCo 与 Motrix owner YAML。 |
| Sharpa in-hand | PPO、APPO HORA teacher、HORA distillation | Sharpa owner YAML 为 PPO/APPO teacher 路径已提交；student distillation 使用 `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`。 |

```{toctree}
:hidden:
:caption: 文档

1-getting_started/0-index
2-user_guide/0-index
3-deployment/0-index
4-developer_guide/0-index
5-reference/0-index
```
