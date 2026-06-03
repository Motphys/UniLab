<h1 align="center"> UniLab </h1>

<h3 align="center">
面向超越 GPU 主导范式的机器人 RL 异构架构
</h3>

<p align="center">语言：简体中文 | <a href="README.md">English</a></p>

<p align="center">
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/arxiv-2605.30313-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/"><img src="https://img.shields.io/badge/paper-UniLab-orange" alt="Paper"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="https://discord.gg/EPCuguRmGX"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab 预告图" width="95%">
</p>

<p align="center"><em>无需 GPU 仿真后端即可训练机器人 RL。预告图由 MotrixSim 渲染。</em></p>

从下面的 `快速演示` 开始运行主训练命令。推荐使用 `uv` 安装；Conda 和 pip 用户目前也应继续遵循 `uv` 工作流。平台相关说明与当前边界见 [安装指南](https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/2-installation.html)。

## ✨ 亮点

```
┌───────────────────┐                            ┌─────────────────────────┐
│  CPU Physics Sim  │   Unified Shared Memory    │   GPU Policy Training   │
│   MuJoCo/Motrix   │ ─────────────────────────▶ │     PPO / SAC / TD3     │
│ Multithread Step  │    SharedReplayBuffer      │ CUDA / MPS / ROCm / XPU │
└───────────────────┘                            └─────────────────────────┘
```

- **异构 RL 运行时：** CPU 并行仿真通过共享内存流式传输 transition，而策略学习运行在 GPU 加速器上。
- **两套物理后端：** MuJoCoUni 和 MotrixSim 通过后端专用适配器和任务 owner 配置接入。
- **统一训练 CLI：** `uv run train` 和 `uv run eval` 覆盖 PPO、MLX PPO、APPO、SAC、TD3 和 FlashSAC；额外的 HORA 与 HIM-PPO 路径以脚本级工作流文档化。
- **配置拥有的任务：** Hydra owner YAML 会同时选择 task、reward、backend 和 algorithm；后端切换通过 `task=<task>/<backend>` 表达。
- **跨平台安装路径：** 仓库覆盖 Linux CUDA、Linux ROCm、Linux XPU，以及 Apple Silicon / macOS 的安装流程。

## 🚀 快速演示

<table>
  <tr>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/dance.jpg" alt="dance demo" width="100%">
      <br>
      <sub><b>dance</b><br>G1 动作跟踪</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/wallflip.jpg" alt="wallflip demo" width="100%">
      <br>
      <sub><b>wallflip</b><br>G1 空翻</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/teaser.jpg" alt="teaser demo" width="100%">
      <br>
      <sub><b>teaser</b><br>MotrixSim 预告图</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/boxtracking.jpg" alt="boxtracking demo" width="100%">
      <br>
      <sub><b>boxtracking</b><br>G1 箱体跟踪</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/inhandgrasp.jpg" alt="inhandgrasp demo" width="100%">
      <br>
      <sub><b>inhandgrasp</b><br>Sharpa 手内抓取</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/sphinx/source/_static/demos/locomani.jpg" alt="locomani demo" width="100%">
      <br>
      <sub><b>locomani</b><br>Go2 移动操作</sub>
    </td>
  </tr>
</table>

```bash
# 0. 如果还没有安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 安装依赖
# 请按你的平台选择对应的安装命令。

# Linux CUDA 或 macOS
make setup-motrix
# 不使用 shell completion 设置时：uv sync --extra motrix
# 如果没有安装 `make`：uv sync --extra motrix && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu

# 3. 预训练 checkpoint 回放（首次运行会从 Hugging Face 下载）
uv run demo dance
```

可用的 demo 名称：`teaser`、`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。
完整的命令与参数请参阅 [统一 CLI](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/1-cli_reference.html) 页面。

> 中国大陆用户：动作、场景和 demo checkpoint 首次运行时会从 Hugging Face 拉取。如果 `huggingface.co`
> 无法访问，请在运行 demo 命令前先将客户端切到社区镜像：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

用于训练与评估：

```bash
uv run train --algo appo --task go2_joystick_flat --sim motrix

uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1

# Linux / 服务器环境下的 Motrix 无头视频导出
uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

这会路由到 `go2_joystick_flat/motrix` 任务 owner 配置，并保持后端选择显式化。

在 macOS / MacBook 上，UniLab CLI 在需要时会通过 `mxpython` 路由 Motrix 交互式回放。Motrix 默认使用交互式回放；要导出无头视频请使用 `--render-mode record`，要跳过回放请使用 `--render-mode none`。更细的脚本级命令请参阅 [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)。

## 🏃 示例运行

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

```bash
uv run train --algo sac --task g1_motion_tracking --sim motrix
```

```bash
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora
```

> Grasp cache 首次训练时会自动从 Hugging Face (`unilabsim/unilab-caches`) 下载到 `src/unilab/assets/caches/`，无需手动操作；如需为自定义 scale 重新生成（较慢）：
>
> ```bash
> bash scripts/sharpa_collect_grasps.sh 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
> ```

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim motrix
uv run eval --algo ppo --task go2_arm_manip_loco --sim motrix --load-run -1
```

使用 `uv run train` 进行训练，使用 `uv run eval` 进行检查点回放，`uv run demo` 用于本地 demo 预设。这些命令可以明确指定算法、任务和后端。

更多训练命令、脚本级入口、算法矩阵、续训流程以及 W&B 细节请参阅 [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)。

## 📚 文档

请使用已发布的 [UniLab 文档](https://unilabsim.github.io/UniLab-doc/)；中文文档入口见 [中文文档索引](https://unilabsim.github.io/UniLab-doc/zh_CN/0-index.html)。高信号入口如下：

- [快速上手](https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/0-index.html)：安装、Docker 运行时、依赖配置和首次运行命令
- [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)：训练、回放、续训流程、Hydra override 和 W&B
- [仿真后端](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/3-backends/0-index.html)：生成的 MuJoCo / Motrix 支持矩阵
- [开发者指南](https://unilabsim.github.io/UniLab-doc/zh_CN/4-developer_guide/0-index.html)：契约、分层与验证边界
- [ADR 索引](https://unilabsim.github.io/UniLab-doc/adr/ADR-0000-index.html)：已采纳的架构决策

## 💬 社群交流

海外用户可加入我们的 [Discord 服务器](https://discord.gg/EPCuguRmGX) 交流与获取帮助。

| 微信群 | 小助手微信 |
| :---: | :---: |
| <img src="docs/sphinx/source/_static/assets/unilab-wechat-group.jpg" alt="UniLab 微信群二维码" width="220"> | <img src="docs/sphinx/source/_static/assets/unilab-wechat-assistant.jpg" alt="UniLab 小助手微信二维码" width="150"> |
| 扫码加入 UniLab 微信群。 | 如果微信群已满，请添加小助手微信，并备注 `unilab交流`。 |

## 🧾 引用

### UniLab

```bibtex
@article{jia2026unilab,
  title         = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author        = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal       = {arXiv preprint arXiv:2605.30313},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30313}
}
```

### 物理后端

```bibtex
@article{jia2026mujocouni,
  title  = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year   = {2026}
}

@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}
```
