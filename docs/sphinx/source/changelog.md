---
orphan: true
---

# Changelog / 变更日志

UniLab follows [Semantic Versioning](https://semver.org/). This shared page
records notable PyPI releases and unreleased changes in English and Chinese;
release dates are PyPI upload dates. For the complete commit history, see the
[UniLab repository](https://github.com/unilabsim/UniLab).

UniLab 遵循[语义化版本](https://semver.org/)。本共享页面以中英文记录重要的
PyPI 版本变更与未发布变更；发布日期采用 PyPI 上传日期。完整提交历史请参阅
[UniLab 仓库](https://github.com/unilabsim/UniLab)。

## 1.3.0 (2026-09-14)

### Breaking changes / 破坏性变更

- The MuJoCo physics executor now uses the published `mjbatch-uni~=0.2.0`
  package instead of `mujoco-uni-runtime`/`mujoco_uni`
  ([#1552](https://github.com/Motphys/UniLab/issues/1552),
  [#1553](https://github.com/Motphys/UniLab/issues/1553)). The `mujoco`
  extra is aligned with MuJoCo 3.11 and checks for the `mjbatch` module at
  runtime. Chunk/forward-sizing knobs and the old Makefile bootstrap targets
  were removed because scheduling now belongs to `mjbatch`. Numerical
  equivalence with the previous executor is not guaranteed; the accepted Go2
  drift is recorded by the #1554 baseline. The backend remains Linux/macOS
  only because `mjbatch` has no Windows wheels.
  MuJoCo 物理执行器改用已发布的 `mjbatch-uni~=0.2.0`，不再使用
  `mujoco-uni-runtime`/`mujoco_uni`（#1552、#1553）。`mujoco` extra 对齐
  MuJoCo 3.11，并在运行时检查 `mjbatch` 模块。chunk/forward 调整项和旧
  Makefile 引导目标已删除，调度改由 `mjbatch` 负责。不保证与旧执行器数值
  等价；可接受的 Go2 漂移由 #1554 基线记录。由于 `mjbatch` 没有 Windows
  wheel，该后端仍仅支持 Linux/macOS。

- The legacy UniLab DomainRandomization provider protocol is removed
  ([#1563](https://github.com/Motphys/UniLab/issues/1563),
  [#1567](https://github.com/Motphys/UniLab/issues/1567)). Fixed model identity
  is now construction-time state; reset and interval randomization use
  backend-owned event terms, `ResetStateTransaction`, and the public UniSim
  plan contract. The `unilab.dr` namespace and provider-side compatibility
  surface are gone.
  移除 UniLab 旧版 DomainRandomization provider 协议（#1563、#1567）。固定
  模型身份改为 construction-time 状态；reset 与 interval 随机化使用后端拥有的
  event term、`ResetStateTransaction` 和公开 UniSim plan contract。
  `unilab.dr` namespace 与 provider 侧兼容面已删除。

### Added / 新增

- Task-owned fixed model/tool variants can differ per environment and are
  materialized through the UniSim construction-time plan contract
  ([#1568](https://github.com/Motphys/UniLab/issues/1568)). The SimToolReal
  representative workload covers fixed-tool rollout parity, reset-time
  mass/inertia randomization, CPU/MJWarp parity, and one PPO learning
  iteration.
  task-owned fixed model/tool variant 现在可以按环境不同，并通过 UniSim
  construction-time plan contract 物化（#1568）。SimToolReal 代表性工作负载覆盖
  fixed-tool rollout、reset-time 质量/惯量随机化、CPU/MJWarp parity 以及一次
  PPO learning iteration。

### Fixed / 修复

- Reward managers no longer emit duplicate `Episode_Reward/*` reset extras;
  the existing episode log remains the source for those metrics
  ([#1570](https://github.com/Motphys/UniLab/issues/1570)).
  Reward manager 不再在 reset extras 中重复输出 `Episode_Reward/*`；既有
  episode log 仍是这些指标的数据源（#1570）。

### Changed / 变更

- The MuJoCo executor dependency is pinned to `mjbatch-uni~=0.2.1` and
  `unisim-core` moves to `>=1.4.0`
  ([unisim#71](https://github.com/unilabsim/unisim/issues/71),
  [mjbatch_uni#28](https://github.com/unilabsim/mjbatch_uni/issues/28)).
  unisim-core 1.4.0 adds per-environment gravity reset, the cross-backend
  body wrench/torque contract, and per-substep callback wrenches through
  `PreStepControlOutput`; on the MuJoCo pre-step control path the executor's
  split-substep sensor copyout now refreshes tracked body state at every
  substep boundary (about 14x faster than the previous host-side recompute,
  with bit-identical trajectories). Tests that pinned the previous executor
  callback protocol and Isaac worker diagnostic text were updated to the
  published contracts.
  MuJoCo 执行器依赖固定为 `mjbatch-uni~=0.2.1`，`unisim-core` 升级到
  `>=1.4.0`（unisim#71、mjbatch_uni#28）。unisim-core 1.4.0 新增逐环境
  重力 reset、跨后端 body wrench/torque 契约，以及通过
  `PreStepControlOutput` 的逐子步 callback wrench；MuJoCo pre-step control
  路径改用执行器的 split-substep 传感器增量拷出，在每个子步边界刷新
  tracked body state（相比此前主机端重算约 14 倍加速，轨迹逐位一致）。
  原先固定旧执行器 callback 协议与 Isaac worker 诊断文案的测试已更新到
  已发布契约。

## 1.2.0 (2026-09-10)

### Breaking changes / 破坏性变更

- Playback overlays now use typed per-environment `DebugPrimitive` values;
  the old `extra_data_getter` marker-array API is removed.
  回放 overlay 改用类型化 per-environment `DebugPrimitive`；旧的
  `extra_data_getter` marker-array API 已移除。

- **Removed downstream-specific tasks.** Go2 arm manipulation/HIM-PPO moved to
  [legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab),
  and Sharpa in-hand/HORA moved to the dedicated `sharpa_rl_unilab` repository
  ([#1547](https://github.com/Motphys/UniLab/issues/1547)). Their task configs,
  assets, scripts, compatibility imports, and core-owned helpers were removed
  from UniLab. The historical 1.2.0 label is retained, but these removals are
  compatibility-breaking; consumers should use the coordinated external
  repository commits.
  **移除下游专属任务。** Go2 机械臂操作/HIM-PPO 迁移到
  [legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab)，
  Sharpa in-hand/HORA 迁移到专用 `sharpa_rl_unilab` 仓库（#1547）。相关任务
  配置、资产、脚本、兼容导入和核心内专属 helper 已从 UniLab 移除。历史版本号
  1.2.0 保持不变，但这些移除具有兼容性破坏；使用方应固定配套的外部仓库提交。

### Added / 新增

- **SuperDex integration.** Added the optional `superdex` extra, initial
  FR3 joint-target and Go2 joystick owners, native persistent CPU workers with
  rank-owned CPU partitions, and the FR3 bot asset download from Hugging Face
  ([#1533](https://github.com/Motphys/UniLab/issues/1533),
  [#1534](https://github.com/Motphys/UniLab/issues/1534)). Published
  SuperDex wheels replace the temporary source build on CPython 3.12/3.13
  Linux x86_64. FR3 playback is interactive-only because its `.superdex_bot`
  has no MJCF visual model; Go2 also supports recorded video playback.
  **SuperDex 集成。** 新增可选 `superdex` extra、首批 FR3 joint-target 与
  Go2 joystick owner、带 rank-owned CPU partition 的原生持久 CPU worker，以及
  来自 Hugging Face 的 FR3 bot 资产下载（#1533、#1534）。在 CPython
  3.12/3.13 Linux x86_64 上，已发布的
  SuperDex wheel 取代临时源码编译。FR3 资产 `.superdex_bot` 没有 MJCF visual
  model，因此只支持交互回放；Go2 另支持录制视频回放。

- **Third-party owner contracts.** Entity/reset transactions support additional
  geom and joint-parameter writes, explicitly bound fixed mocap bodies, and
  versioned training-progress export/import for downstream community packages
  such as Wuji. Playback uses typed `DebugPrimitive` overlays and exposes an
  embeddable `SnapshotPlaybackSession` for custom evaluation loops.
  **第三方 owner contract。** Entity/reset transaction 支持更多 geom 与 joint
  参数写入、显式绑定的固定 mocap body，以及供 Wuji 等下游社区包使用的版本化
  training-progress 导出/导入。回放改用类型化 `DebugPrimitive` overlay，并暴露可
  嵌入自定义评估循环的 `SnapshotPlaybackSession`。

### Changed / 变更

- **Compatibility updates.** Requires `unisim-core>=1.2.0`, pins
  `unilab-rl==1.2.0`, and updates the ROCm lock profile. This carries the
  shared backend, collector, and playback contracts used by the new owners.
  Because the SuperDex extra can also bring plain MuJoCo, `sim=mujoco` now
  checks the `mujoco_uni` runtime binding.
  **兼容性更新。** 要求 `unisim-core>=1.2.0`，固定 `unilab-rl==1.2.0`，并更新
  ROCm lock profile。这承载新 owner 所需的共享 backend、collector 与 playback
  contract。由于 SuperDex extra 也可能带入普通 MuJoCo，`sim=mujoco` 现在会检查
  `mujoco_uni` runtime binding。

### Fixed / 修复

- PPO action-standard-deviation logging now uses the current policy
  distribution, including state-dependent action scales. Nested play-profile
  merging also preserves unrelated fields and sibling observation terms when a
  partial override is applied.
  PPO action standard deviation 日志现在读取当前 policy distribution，可正确覆盖
  state-dependent action scale。嵌套 play profile 合并时，局部 override 也会保留
  未相关字段和 sibling observation term。

## 1.1.0 (2026-09-06)

- Added the Newton backend owner path and native ViewerGL playback, including
  the initial G1 walk PPO/SAC owners
  ([#1509](https://github.com/Motphys/UniLab/issues/1509),
  [#1511](https://github.com/Motphys/UniLab/issues/1511)).
  新增 Newton backend owner 路径与原生 ViewerGL 回放，包括首批 G1 walk
  PPO/SAC owner（#1509、#1511）。

- Fixed multi-GPU routing for spawned data-parallel collectors: rank-local
  backend devices are selected before CUDA/runtime initialization, including
  Genesis on non-zero physical GPUs and Newton’s rank-local device override.
  修复 spawn data-parallel collector 的多 GPU 路由：在 CUDA/runtime 初始化前选择
  rank-local backend 设备，包括非零物理 GPU 上的 Genesis 以及 Newton 的
  rank-local device override。

- Added third-party task-package discovery through the `unilab.tasks` entry
  point group and generic interval domain-randomization dispatch
  ([#1501](https://github.com/Motphys/UniLab/issues/1501),
  [#1504](https://github.com/Motphys/UniLab/issues/1504)).
  新增通过 `unilab.tasks` entry-point group 发现第三方 task package 的能力，
  以及通用 interval domain-randomization dispatch（#1501、#1504）。

- Updated the shared runtime contracts: `unisim-core>=1.1.3`, MuJoCo/MJWarp
  extras on the MuJoCo 3.11 line, `mujoco-uni-runtime==0.5.0`, and the matching
  ROCm profile. Go1 mesh assets also moved to the shared Hugging Face asset
  flow and out of the published wheel/sdist.
  更新共享 runtime contract：`unisim-core>=1.1.3`、MuJoCo/MJWarp extra 对齐
  MuJoCo 3.11、`mujoco-uni-runtime==0.5.0` 以及配套 ROCm profile。Go1 mesh
  资产同时迁移到共享 Hugging Face asset 流程，并从发布的 wheel/sdist 移除。

- Expanded the bilingual backend evidence, installation, sim-to-sim, and
  project-rationale documentation.
  扩展了 backend 证据、安装、sim-to-sim 与项目定位的双语文档。

## 1.0.0 (2026-09-04)

- Promoted the 0.1 architecture to the first stable package line: consume the
  production `unilab-rl==1.0.0` release instead of the 0.2 TestPyPI line,
  broaden the exact Torch pins to ranges below 2.12 (with the Linux aarch64
  floor at 2.9), and publish canonical project metadata and links.
  将 0.1 架构提升为首个稳定包版本线：消费生产版 `unilab-rl==1.0.0` 而不是
  0.2 TestPyPI 版本，将精确 Torch 钉版放宽到低于 2.12 的范围（Linux aarch64
  下限为 2.9），并发布规范项目元数据与链接。

- Narrowed the packaged task surface: non-production MicroDuck and T800 task
  families moved to downstream ecosystem packages, and unused large motion files
  were removed. The manager-based runtime and production task owners introduced
  in 0.1.0 remain the supported API.
  收窄打包任务面：非生产 MicroDuck 与 T800 task family 迁移到下游生态包，
  并移除未使用的大 motion 文件。0.1.0 引入的 manager-based runtime 与生产
  task owner 仍是受支持 API。

- Fixed event-curriculum velocity updates to re-read the live velocity range and
  fail closed when newly activated angular axes were not bound at construction.
  Reset terms can also read root poses already staged in the same transaction.
  修复 event-curriculum velocity 更新：每次事件应用都会重新读取当前 velocity
  range；构造期未绑定的角度轴被激活时会 fail closed。reset term 也可以读取同一
  transaction 中已 stage 的 root pose。

- Added tag-based trusted PyPI publishing with tag/version and successful-CI
  checks, distribution metadata validation, wheel installation, and an import
  smoke test.
  新增基于 tag 的 PyPI trusted publishing，包含 tag/版本与成功 CI 校验、发行
  元数据检查、wheel 安装和导入 smoke test。

## 0.1.0 (2026-09-04)

- First functional PyPI package. UniLab moved to the NumPy Manager-Based API
  with Hydra task owners for observations, actions, rewards, terminations,
  events, commands, curricula, and metrics; legacy monolithic environments were
  removed
  ([#1042](https://github.com/Motphys/UniLab/issues/1042)).
  首个可实际使用的 PyPI 包。UniLab 迁移到 NumPy Manager-Based API，使用 Hydra
  task owner 声明 observation、action、reward、termination、event、command、
  curriculum 与 metrics；旧的一体化 environment 已移除（#1042）。

- Physics adapters moved behind `unisim-core>=0.1.14` (`unisim`) with MuJoCo,
  Motrix, MJWarp, Drake, Genesis, IsaacGym, and IsaacSim identities. RL
  algorithms, runners, collectors, IPC, and training logs moved behind
  `unilab-rl==0.2.0` (`uni_rl`), preserving the package boundary that prevents
  learner code from importing UniLab.
  物理适配器迁移到 `unisim-core>=0.1.14`（`unisim`），覆盖 MuJoCo、Motrix、
  MJWarp、Drake、Genesis、IsaacGym 与 IsaacSim。RL 算法、runner、collector、
  IPC 与训练日志迁移到 `unilab-rl==0.2.0`（`uni_rl`），保持 learner 代码不得
  导入 UniLab 的包边界。

- The unified `train`, `eval`, and `demo` CLI covers PPO, APPO, SAC, TD3, and
  FlashSAC workflows; robot/scene/motion assets and demo checkpoints download
  from Hugging Face on first use. HORA and HIM-PPO remained script-level
  workflows at this release.
  统一 `train`、`eval` 与 `demo` CLI 覆盖 PPO、APPO、SAC、TD3 与 FlashSAC
  工作流；robot/scene/motion 资产和演示 checkpoint 首次使用时从 Hugging Face
  下载。HORA 与 HIM-PPO 在该版本仍是脚本级工作流。

## 0.0.0 (2026-05-30)

- Placeholder release used only to reserve the `unilab` PyPI name. It contained
  no UniLab runtime code and declared no dependencies.
  仅用于保留 `unilab` PyPI 名称的占位发布，不包含 UniLab runtime 代码，也没有
  声明依赖。
