---
orphan: true
---

# Changelog / 变更日志

UniLab follows [Semantic Versioning](https://semver.org/). This shared page
records notable releases in English and Chinese; for the day-to-day commit log,
see the [UniLab repository](https://github.com/unilabsim/UniLab).

UniLab 遵循[语义化版本](https://semver.org/)。本共享页面以中英文记录重要版本变更；
日常提交记录请参阅 [UniLab 仓库](https://github.com/unilabsim/UniLab)。

## Unreleased / 未发布

- Update the required `unisim-core` release to `>=1.2.0` and the pinned
  `unilab-rl` release to `==1.2.0`, including the ROCm profile.
  将必需的 `unisim-core` 版本更新为 `>=1.2.0`，并将钉定的 `unilab-rl` 版本更新为
  `==1.2.0`，ROCm 配置档同步更新。

- Host the FR3 SuperDex native bot assets (collision SDF, render GLB, license)
  on the Hugging Face dataset
  [unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots),
  consistent with the other robot mesh assets. The asset hub downloads the
  `bots/arms/fr3_v2` snapshot into `src/unilab/assets/` on first use, and
  `uv run unilab-pull-assets --robot fr3_v2` pre-fetches it;
  `SUPERDEX_ASSETS_PATH` / `env.superdex_assets_root` remain as overrides for
  auditing a local `project_superdex` checkout and take precedence without
  downloading.
  FR3 SuperDex 原生 bot 资产（collision SDF、render GLB、许可证）改为托管在
  Hugging Face 数据集
  [unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots)，
  与其他机器人 mesh 资产的处理方式一致。asset hub 首次使用时自动把
  `bots/arms/fr3_v2` 快照下载到 `src/unilab/assets/`，也可用
  `uv run unilab-pull-assets --robot fr3_v2` 预拉取；
  `SUPERDEX_ASSETS_PATH` / `env.superdex_assets_root` 保留为审计本地
  `project_superdex` checkout 的覆盖方式，优先级更高且不会触发下载。

- Add a `superdex` optional extra (`uv sync --extra superdex` or
  `pip install "unilab[superdex]"`) that pulls the published SuperDex
  Physics/Robotics 1.0.0 wheels (CPython 3.12/3.13, Linux x86_64) through
  `unisim-core[superdex]`, replacing the temporary source-build setup. The
  `sim=mujoco` runtime check now guards on the `mujoco_uni` binding, since
  plain `mujoco` also arrives with the superdex extra.
  SuperDex native interactive playback crashing on the first rendered frame
  (`eval --sim superdex --render-mode interactive`) was first fixed in
  `unisim-core` 1.1.6; the required release line is now `>=1.2.0`.
  新增 `superdex` optional extra（`uv sync --extra superdex` 或
  `pip install "unilab[superdex]"`），通过 `unisim-core[superdex]`
  安装已发布的 SuperDex Physics/Robotics 1.0.0 wheel（CPython 3.12/3.13、
  Linux x86_64），取代临时的源码编译安装方式。`sim=mujoco` 的运行时检查
  改为检查 `mujoco_uni` 绑定，因为普通 `mujoco` 包也会随 superdex extra
  装入。SuperDex 原生 interactive 回放首帧渲染崩溃的问题
  （`eval --sim superdex --render-mode interactive`）最早在 `unisim-core`
  1.1.6 修复；当前必需的版本线为 `>=1.2.0`。

- Go2 arm manipulation/locomotion, its legacy helpers, assets, tools and HIM-PPO
  owners moved to [legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab)
  under [#1528](https://github.com/unilabsim/UniLab/issues/1528). Dedicated source
  content and compatibility imports are removed; shared backend contracts remain.
  Go2 机械臂任务及专属辅助模块、资产、工具和 HIM-PPO 配置迁入独立仓库，原入口移除。
  Per maintainer instruction, this migration does not change repository versions
  or publish a release; consumers use the coordinated migration commits.
  按维护者要求，本次不修改仓库版本号、不发布新版本，消费方固定配套迁移提交。

## 1.1.0 (2026-09-06)

- Update the required `unisim-core` release to `>=1.1.3`, including the ROCm
  profile, so UniLab consumes the current shared physics-backend contract.
  将必需的 `unisim-core` 版本更新为 `>=1.1.3`，并同步更新 ROCm 配置档，使 UniLab
  使用当前共享的物理后端 contract。
- Add the Newton backend owner path and native ViewerGL playback integration,
  with explicit device routing for spawned collectors.
  增加 Newton backend owner 路径和原生 ViewerGL 回放集成，并为 spawn collector
  增加显式设备路由。
- Align the MuJoCo extras on the 3.11 line and document the version-switch
  fallback path for the native runtime extension.
  将 MuJoCo extras 统一到 3.11 版本线，并补充原生 runtime 扩展切换版本时的回退路径
  文档。
- Add third-party task-package discovery through the `unilab.tasks` entry-point
  group and generic interval domain-randomization dispatch.
  增加通过 `unilab.tasks` entry-point group 发现第三方 task package 的能力，并增加通用
  interval domain-randomization dispatch。
- Expand the bilingual documentation around backend support evidence, platform
  setup, sim-to-sim contracts, and the Why UniLab project rationale.
  扩展双语文档，覆盖 backend 支持证据、平台安装、sim-to-sim contract 和 Why UniLab
  项目定位。

## 1.0.0

- `pyproject.toml` declares package version `1.0.0`. The Manager-Based API migration
  (roadmap #1042) is merged: manager core and NumPy term library ported from mjlab
  1.6.0, all production tasks on the Manager-Based runtime, and legacy monolithic
  envs removed.
  `pyproject.toml` 声明 package 版本 `1.0.0`。Manager-Based API 迁移（roadmap #1042）
  完成：manager core 和 NumPy term library 从 mjlab 1.6.0 移植，所有 production task
  运行在 Manager-Based runtime 上，并移除旧的 monolithic env。
- Physics backends live in the separate `unisim-core` package (MuJoCo, Motrix,
  mjwarp, IsaacGym, IsaacSim, Genesis, Drake); RL algorithms and the async runtime
  live in the separate `unilab-rl` package (`uni_rl`).
  物理后端位于独立的 `unisim-core` package（MuJoCo、Motrix、mjwarp、IsaacGym、IsaacSim、
  Genesis、Drake）；RL 算法和异步 runtime 位于独立的 `unilab-rl` package（`uni_rl`）。
- Robot meshes/textures are hosted on the Hugging Face dataset
  `unilabsim/unilab-robots` and pulled on demand. The bilingual Sphinx source
  layout is documented in `docs/sphinx/README.md`.
  机器人 mesh/texture 托管在 Hugging Face 数据集 `unilabsim/unilab-robots`，按需拉取。
  双语 Sphinx 源码布局见 `docs/sphinx/README.md`。
- ADR, glossary, and changelog pages are shared content rather than per-language
  pages.
  ADR、术语表和 changelog 页面采用共享内容，而不是分别维护语言版本。

## 0.1.0

- `pyproject.toml` declares package version `0.1.0` and the first-level console
  entrypoints `train`, `eval`, `demo`, `unilab-complete`, `unilab-viz-nan`, and
  `unilab-export-scene`.
  `pyproject.toml` 声明 package 版本 `0.1.0`，并提供首批顶层 console entrypoint：
  `train`、`eval`、`demo`、`unilab-complete`、`unilab-viz-nan` 和 `unilab-export-scene`。
- The repository README documents the CPU simulation, shared-memory runtime, and
  GPU learning architecture, with MuJoCo and Motrix named as physics backends.
  仓库 README 介绍 CPU 仿真、共享内存 runtime 和 GPU learning 架构，并将 MuJoCo 与
  Motrix 列为物理后端。
- Accepted ADRs in `docs/sphinx/source/adr/README.md` cover runtime layer
  boundaries, backend capability boundaries, task owner config composition,
  registry bootstrap, and observation / IPC contracts.
  `docs/sphinx/source/adr/README.md` 中的已接受 ADR 覆盖 runtime 分层边界、backend
  capability 边界、task owner 配置组合、registry bootstrap 以及 observation/IPC contract。
