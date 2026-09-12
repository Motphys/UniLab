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

- Replace the `mujoco-uni-runtime` dependency (`mujoco_uni` import) with the
  `mjbatch` native batch engine across the repository (roadmap
  [#1552](https://github.com/unilabsim/UniLab/issues/1552),
  [#1553](https://github.com/unilabsim/UniLab/issues/1553)). The `mujoco`
  extra now installs `mujoco~=3.11.0` plus `mjbatch` pinned to the
  [integration fork](https://github.com/unilabsim/mjbatch); the fork's final
  distribution identity (PyPI package vs git pin, and prebuilt wheels) is the
  roadmap's open maintainer item. The `sim=mujoco` CLI runtime check now gates
  on the `mjbatch` module. A post-swap ablation slimmed the pinned fork's
  API ([#1557](https://github.com/unilabsim/UniLab/issues/1557)): the
  per-substep callback is `fn(k, state, ctrl)` (no `callback_sensordata`
  argument), `steps_done` / `stop_on_warning` are gone, and the hfield
  scanner is height-only with its own validation. Numerical equivalence
  before and after the swap is **not** guaranteed; the accepted drift is
  characterized by the #1554 drift baseline.
  全仓库将 `mujoco-uni-runtime` 依赖（`mujoco_uni` 导入）替换为 `mjbatch`
  原生 batch 引擎（roadmap #1552、#1553）。`mujoco` extra 现安装
  `mujoco~=3.11.0` 加钉住的 [集成 fork](https://github.com/unilabsim/mjbatch)
  `mjbatch`；fork 的最终分发身份（PyPI package 还是 git 钉版、是否提供预编译
  wheel）是 roadmap 上的待定维护事项。`sim=mujoco` 的 CLI 运行时检查改为检查
  `mjbatch` 模块。替换后的消融精简了钉住 fork 的 API（#1557）：per-substep
  回调为 `fn(k, state, ctrl)`（不再有 `callback_sensordata` 参数），
  `steps_done` / `stop_on_warning` 已移除，hfield 扫描器只输出高度并自带
  校验。替换前后数值不保证一致；接受的漂移由 #1554 漂移基线表征。

- Deprecate and remove the MuJoCo chunk/forward knobs: `EnvCfg` fields
  `post_step_forward_sensor`, `adaptive_chunk_size`, and `chunk_size`, the
  `bench_nsteps` backend kwarg, the matching Hydra owner keys, and the
  `make mujoco MJ=<version>` / `check-cxx-toolchain` / `setup-mujoco` Makefile
  targets are gone. `mjbatch` schedules per-simulation work without a chunk
  knob, and step ends one substep behind the state by default, matching the
  previous `post_step_forward_sensor=False` semantics; the per-env model
  variant machinery (`ModelVariantSpec` materialization on the MuJoCo backend)
  is no longer supported there — init-lifecycle geometry overrides move to the
  remaining variant-capable backends. Windows support is unchanged: the
  MuJoCo physics backend stays Linux/macOS-only because `mjbatch` ships no
  Windows wheels.
  弃用并移除 MuJoCo chunk/forward 旋钮：`EnvCfg` 字段
  `post_step_forward_sensor`、`adaptive_chunk_size`、`chunk_size`、`bench_nsteps`
  后端 kwarg、对应的 Hydra owner 键，以及 Makefile 目标
  `make mujoco MJ=<version>` / `check-cxx-toolchain` / `setup-mujoco` 均已删除。
  `mjbatch` 在没有 chunk 旋钮的情况下调度 per-simulation 工作，且 sensordata
  默认落后一个子步，与之前的 `post_step_forward_sensor=False` 语义一致；
  MuJoCo 后端不再支持 per-env 模型 variants（`ModelVariantSpec`
  materialization）——init-lifecycle 几何覆盖改由仍支持 variants 的后端提供。
  Windows 支持不变：由于 `mjbatch` 不提供 Windows wheel，MuJoCo 物理后端
  仍然只支持 Linux/macOS。

## 1.2.0 (2026-09-10)

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

- The FR3 SuperDex owner defaults play to the native interactive viewer
  (`play_render_mode=interactive`, `play_env_num=1`). Record (video) playback
  stays unavailable for the `.superdex_bot` asset, which carries no MJCF
  visual model; use `training.play_render_mode=none` for headless runs.
  FR3 SuperDex owner 的 play 默认改为 native interactive viewer
  （`play_render_mode=interactive`，`play_env_num=1`）。`.superdex_bot` 资产没有
  MJCF visual model，record（视频）回放仍不可用；无显示环境使用
  `training.play_render_mode=none`。

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
