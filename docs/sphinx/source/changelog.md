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

## 1.5.4 (2026-10-31)

### Changed / 变更

- APPO, PPO, SAC, and FlashSAC now accumulate action, control, reward,
  observation, command, and metrics finite-validation verdicts on device and
  poll them every 32 steps by default, avoiding per-step GPU/CPU
  synchronization. NanGuard physics snapshots also default to every 32 steps;
  nonfinite values may therefore be reported at the next validation boundary.
  The documented debug overrides restore immediate per-step validation
  ([#2554](https://github.com/Motphys/UniLab/pull/2554)).
  APPO、PPO、SAC 与 FlashSAC 现在在设备端累积 action、control、reward、
  observation、command 与 metrics 的有限值校验结果，并默认每 32 步轮询一次，
  避免逐步 GPU/CPU 同步。NanGuard 物理状态快照也默认每 32 步采集一次；因此
  非有限值可能到下一个校验边界才会报告。文档中的调试覆盖项可恢复逐步即时
  校验（[#2554](https://github.com/Motphys/UniLab/pull/2554)）。
- The default and ROCm profiles now pin `unisim-core==2.2.5`, including the
  Motrix and SuperDex extras
  ([UniSim #492](https://github.com/unilabsim/unisim/pull/492)).
  默认与 ROCm profile 现在均钉定 `unisim-core==2.2.5`，包括 Motrix 与
  SuperDex extra（[UniSim #492](https://github.com/unilabsim/unisim/pull/492)）。

### Fixed / 修复

- Motrix whole-MJCF rotated-root resets now preserve body-frame generalized
  angular velocity on full, tensor, and selected-row `qvel` reads through
  UniSim 2.2.5; world-frame angular sensors remain unchanged
  ([UniSim #492](https://github.com/unilabsim/unisim/pull/492)).
  通过 UniSim 2.2.5，Motrix whole-MJCF 旋转根节点 reset 现在在完整、tensor 与
  选中行 `qvel` 读回中保持 body-frame 广义角速度；world-frame 角速度传感器
  保持不变（[UniSim #492](https://github.com/unilabsim/unisim/pull/492)）。
- Standalone absolute-path demo checkpoints without `run_config.json` now use
  the explicit no-source-run playback path, and rsl-rl checkpoints load onto
  the selected playback device. This restores `uv run demo dance` on CPU and
  MPS ([#2557](https://github.com/Motphys/UniLab/issues/2557)).
  没有 `run_config.json` 的独立绝对路径 demo checkpoint 现在使用显式的
  no-source-run 回放路径，rsl-rl checkpoint 也会加载到所选回放设备。这恢复了
  CPU 与 MPS 上的 `uv run demo dance`
  （[#2557](https://github.com/Motphys/UniLab/issues/2557)）。
- The macOS teaser now exposes the UniLab package parent to external
  `mxpython` and resolves scene assets in the active uv environment before
  launching the renderer, restoring `uv run demo teaser`
  ([#2555](https://github.com/Motphys/UniLab/issues/2555)).
  macOS teaser 现在会向外部 `mxpython` 暴露 UniLab 包父目录，并在启动渲染器前
  从当前 uv 环境解析场景资产，从而恢复 `uv run demo teaser`
  （[#2555](https://github.com/Motphys/UniLab/issues/2555)）。

## 1.5.3 (2026-10-29)

### Added / 新增

- Manager action terms that require state feedback can now recompute control on
  every physics substep. The Manager refreshes public packed state, reapplies
  the action, and steps the backend once per substep while preserving the
  existing one-control-per-step behavior for feed-forward terms
  ([#2535](https://github.com/Motphys/UniLab/issues/2535)).
  需要 state feedback 的 Manager action term 现在可以在每个 physics substep
  重新计算控制。Manager 会刷新公开 packed state、重新应用 action，并按
  substep 单独推进 backend；feed-forward term 仍保持每步一次控制的既有行为
  （[#2535](https://github.com/Motphys/UniLab/issues/2535)）。

### Changed / 变更

- The public environment contracts moved from `unilab.rl.contracts` to
  `unilab.base.contracts`, and the old namespace is removed. Environment
  construction no longer imports the RL algorithm stack, `unilab.base` can load
  the contracts lazily without UniSim, and training dependencies such as
  TensorBoard, TensorDict, rsl-rl, and W&B are available through the single
  `unirl` extra instead of the base environment install
  ([#2542](https://github.com/Motphys/UniLab/issues/2542)).
  公共 environment contract 从 `unilab.rl.contracts` 移到
  `unilab.base.contracts`，旧命名空间已删除。环境构造不再导入 RL algorithm
  stack；`unilab.base` 可以在不加载 UniSim 的情况下 lazy 加载 contract；
  TensorBoard、TensorDict、rsl-rl 与 W&B 等训练依赖改由单一的 `unirl`
  extra 提供，不再进入基础 environment 安装
  （[#2542](https://github.com/Motphys/UniLab/issues/2542)）。
- Manager reset and interval randomization now consume UniSim's unified
  capability sets instead of negotiating separate host and device planes. The
  default and ROCm profiles pin `unisim-core==2.2.3`, and backend payload and
  device-execution obligations remain fail-closed
  ([#2550](https://github.com/Motphys/UniLab/issues/2550)).
  Manager reset 与 interval randomization 现在消费 UniSim 的统一 capability
  set，不再分别协商 host/device plane。默认与 ROCm profile 均钉定
  `unisim-core==2.2.3`；backend payload 和设备执行义务仍然 fail closed
  （[#2550](https://github.com/Motphys/UniLab/issues/2550)）。
- The unused `options` parameter is removed from
  `ManagerBasedRlEnv.reset()`; passing it now raises the normal Python
  unexpected-keyword `TypeError` without a compatibility layer
  ([#2538](https://github.com/Motphys/UniLab/issues/2538)).
  `ManagerBasedRlEnv.reset()` 删除了未使用的 `options` 参数；传入该参数会按
  Python 常规行为抛出 unexpected-keyword `TypeError`，没有兼容层
  （[#2538](https://github.com/Motphys/UniLab/issues/2538)）。

### Fixed / 修复

- Mapped Entity body selections are now bound to backend body-state rows by
  public body name. Scene-layout body IDs remain model addresses, so packed
  backends may omit the world row without confusing the two address spaces
  ([#2535](https://github.com/Motphys/UniLab/issues/2535)).
  Mapped Entity 的 body 选择现在按公开 body name 绑定到 backend body-state
  row。Scene-layout body ID 继续表示 model 地址，因此 packed backend 可以省略
  world row，而不会混淆两种地址空间
  （[#2535](https://github.com/Motphys/UniLab/issues/2535)）。
- An explicit seed passed to `reset()` is applied before cold-path state
  initialization, so initialization and reset randomization consume the
  requested seed ([#2538](https://github.com/Motphys/UniLab/issues/2538)).
  显式传入 `reset()` 的 seed 现在会在冷路径 state initialization 之前应用，
  initialization 与 reset randomization 都会消费该 seed
  （[#2538](https://github.com/Motphys/UniLab/issues/2538)）。
- Interactive playback now resolves checkpoints and runs sim2sim contract
  preflight before constructing playback environments for rsl-rl, APPO, SAC,
  and FlashSAC sessions
  ([#2539](https://github.com/Motphys/UniLab/issues/2539)).
  Interactive playback 现在会先解析 checkpoint 并执行 sim2sim contract
  preflight，然后再为 rsl-rl、APPO、SAC 与 FlashSAC session 构造 playback
  环境（[#2539](https://github.com/Motphys/UniLab/issues/2539)）。

## 1.5.2 (2026-10-27)

### Added / 新增

- The Manager tensor ray observation now consumes UniSim's full-scene public
  `RayCollisionDescription`, so mesh/hfield materialization no longer requires
  a backend-specific `uni_ray` builder, and entity-local packed pose reads are
  aggregated onto the public full-scene body axis. `RayPatternCfg` supports
  ring, parallel-grid, and pinhole-camera ray patterns; body, yaw-only, and
  world alignment; body or public-site attachment with a local sensor offset;
  `min_distance`; public geom/tag/body/ancestor filters; optional normal and
  hit-point outputs in addition to distance; and reset-boundary per-row geometry
  synchronization. The optional `ray` extra now resolves the published
  `uniray[warp]==0.1.0` distribution in both CUDA and ROCm lockfiles
  ([UniSim #472](https://github.com/unilabsim/unisim/issues/472)).
  Manager tensor ray observation 现在消费 UniSim 公开 full-scene
  `RayCollisionDescription`，mesh/hfield materialization 不再要求 backend-specific
  `uni_ray` builder，并将 entity-local packed pose read 聚合到公开 full-scene body
  轴。`RayPatternCfg` 支持 ring、平行 grid 与 pinhole-camera ray pattern，
  body/yaw-only/world alignment、body/public-site attachment 与本地 sensor
  offset、`min_distance`、公开 geom/tag/body/ancestor filter、可选 normal 与
  hit-point 输出，以及 reset 边界的逐行 geometry 同步。可选 `ray` extra
  在 CUDA 与 ROCm lockfile 中均解析已发布的 `uniray[warp]==0.1.0` 发行版
  （[UniSim #472](https://github.com/unilabsim/unisim/issues/472)）。
  The first facade is a minimal grouped ray sensor, not a camera: intrinsics,
  image projection, noise, history, and scheduling remain application logic.
  Full-scene pose aggregation covers entity-owned bodies plus the identity
  world root; uncovered non-root bodies fail closed. MuJoCo and MJWarp
  fixed-variant collision descriptors now publish per-row geometry through the
  capability added by [UniSim #479](https://github.com/unilabsim/unisim/issues/479);
  Motrix source mesh/hfield scenes follow the UniSim fail-closed boundary.
  The real end-to-end fixture covers
  primitive geometry, static filtering, minimum distance, attachment
  transformation, site attachment, and normals; focused fake-caster tests cover
  grid and pinhole ray geometry, yaw/world alignment, hit-point packing, and
  geometry synchronization; mesh/hfield
  numerical parity is owned by the UniSim and `uni_ray` contract suites.
  首个 facade 是最小 grouped ray sensor，不是 camera：intrinsics、图像投影、
  noise、history 与 scheduling 仍属于应用逻辑。full-scene pose 聚合覆盖
  entity-owned body 与 identity world root；未覆盖的非 root body fail closed。
  MuJoCo 与 MJWarp fixed-variant collision scene 以及 Motrix source
  mesh/hfield scene 遵循 UniSim fail-closed 边界；fixed-variant collision
  descriptor 通过 [UniSim #479](https://github.com/unilabsim/unisim/issues/479)
  新增的能力发布逐行 geometry。真实端到端 fixture 覆盖
  primitive geometry、static filtering、minimum distance、attachment
  transformation、site attachment 与 normal；grid/pinhole ray 几何、yaw/world alignment、hit-point packing 与 geometry synchronization 由 focused fake-caster test 覆盖；mesh/hfield 数值 parity 由 UniSim 与 `uni_ray` contract suite 负责。
- A direct-engine G1 physics-throughput benchmark now covers Genesis
  Metal/AMDGPU/CUDA, optional MJWarp and Newton CUDA, mjbatch CPU, and MotrixSim
  CPU. It shares the stand keyframe and control setup, maps backend-specific
  quaternion and actuator semantics, synchronizes the selected device, and emits
  throughput statistics, JSON, and plots. The tool measures pure physics
  stepping only and does not add a training-runtime or backend-support claim
  ([#2518](https://github.com/Motphys/UniLab/pull/2518)).
  direct-engine G1 physics-throughput benchmark 现在覆盖 Genesis
  Metal/AMDGPU/CUDA、可选 MJWarp 与 Newton CUDA、mjbatch CPU 以及 MotrixSim
  CPU。它共享 stand keyframe 与控制设置，映射 backend 特有的 quaternion 和
  actuator 语义，同步所选设备，并输出吞吐统计、JSON 与图表。该工具只测量纯
  physics stepping，不新增 training-runtime 或 backend 支持承诺
  （[#2518](https://github.com/Motphys/UniLab/pull/2518)）。

### Changed / 变更

- Genesis runtime domain randomization now follows the unified device-resident
  `unisim-core` lifecycle. Manager interval negotiation selects the
  backend-bound device plane, wrench payloads and body IDs enter the interval
  plan on that device, and the packed reset device is declared before cold
  event-term binds. The Genesis G1 walk SAC owner re-enables `pd_gains` with
  the PPO owner because Genesis declares device-plane packed reset `kp`/`kd`;
  host-plane runtime DR, push, velocity-delta, and other undeclared terms
  still fail closed
  ([#2520](https://github.com/Motphys/UniLab/issues/2520)).
  Genesis 运行时域随机化现在遵循 `unisim-core` 统一的 device-resident 生命
  周期。Manager interval 协商选择 backend 绑定的 device plane；wrench payload
  与 body ID 在该设备上进入 interval plan；packed reset 设备会在冷路径
  event-term bind 之前声明。Genesis G1 walk SAC owner 与 PPO owner 一致重新
  启用 `pd_gains`，因为 Genesis 声明了 device-plane packed reset 的
  `kp`/`kd`；host-plane 运行时 DR、push、velocity-delta 以及其他未声明 term
  仍会 fail closed（[#2520](https://github.com/Motphys/UniLab/issues/2520)）。
- The ROCm lock profile now publishes the exact `genesis-world==1.3.3` optional
  extra and resolves its opt-in installation from registry packages. The default
  `make sync-rocm` target still installs only the `mujoco` extra
  ([#2531](https://github.com/Motphys/UniLab/issues/2531)).
  ROCm lock profile 现在发布精确的 `genesis-world==1.3.3` optional extra，并让
  opt-in 安装从 registry package 解析。默认 `make sync-rocm` target 仍然只安装
  `mujoco` extra（[#2531](https://github.com/Motphys/UniLab/issues/2531)）。

### Fixed / 修复

- The CUDA inference IPC ring now uses event synchronization instead of IPC-event
  stream waits on HIP/ROCm. This avoids the ROCm CLR 7.2 emulated-event modulo
  defect that produced `hipErrorInvalidValue` when the 33rd signal-slot event
  was reused, allowing production-shape ROCm Genesis SAC runs to complete. The
  compatibility path is performance-impacting; NVIDIA CUDA continues to use
  stream waits
  ([#2530](https://github.com/Motphys/UniLab/issues/2530)).
  CUDA inference IPC ring 在 HIP/ROCm 上现在使用 event synchronize，而不是
  IPC-event stream wait。这避开了 ROCm CLR 7.2 emulated-event 的 modulo 缺陷；
  该缺陷在第 33 个 signal-slot event 复用时触发 `hipErrorInvalidValue`，修复后
  production-shape ROCm Genesis SAC run 可以正常完成。兼容路径会影响性能；
  NVIDIA CUDA 继续使用 stream wait
  （[#2530](https://github.com/Motphys/UniLab/issues/2530)）。

## 1.5.1 (2026-10-23)

### Added / 新增

- FlashSAC data-parallel learners now capture and replay the whole update
  cycle, including NCCL reductions, by default on supported compiled CUDA
  execution. Runner synchronizer hooks coordinate capture and replay, a
  packed loss-validity sentinel keeps optimizer and target updates rank
  consistent, and DP capture failure fails closed instead of allowing one rank
  to fall back to eager collectives. Set
  `algo.algo_params.dp_whole_cycle_cuda_graph=false` to disable DP capture, or
  `algo.algo_params.use_whole_cycle_cuda_graph=false` to disable capture on all
  ranks. `training.dp_gradient_wire_dtype=bfloat16` additionally opts into
  lossy BF16 communication for sufficiently large FP32 gradient groups; FP32
  remains the default and long-run convergence equivalence has not been
  established
  ([#2499](https://github.com/Motphys/UniLab/pull/2499)).
  FlashSAC 数据并行 learner 在受支持的编译 CUDA 执行路径上，现在默认捕获并回放
  包含 NCCL reduction 的完整 update cycle。Runner synchronizer hook 协调捕获与
  回放；打包的 loss-validity sentinel 保证 optimizer 与 target update 在各 rank
  一致；DP 捕获失败会 fail closed，而不是允许单个 rank 回退到 eager collectives。
  设置 `algo.algo_params.dp_whole_cycle_cuda_graph=false` 可关闭 DP 捕获；
  `algo.algo_params.use_whole_cycle_cuda_graph=false` 可在所有 rank 关闭捕获。
  `training.dp_gradient_wire_dtype=bfloat16` 可额外为足够大的 FP32 gradient
  group 选择有损 BF16 通信；FP32 仍为默认，且尚未证明长时间训练收敛等价
  （[#2499](https://github.com/Motphys/UniLab/pull/2499)）。

### Changed / 变更

- Multi-GPU topology remains explicit and simple: omit
  `CUDA_VISIBLE_DEVICES` for one rank, or provide one entry per selected GPU.
  README now shows minimal two-GPU FlashSAC training and the manually managed
  CUDA MPS flow. An explicit `training.cuda_process_sharing=mps` request
  normalizes numeric GPU entries to host UUIDs before rank derivation and CUDA
  initialization, so the documented per-GPU MPS launch no longer fails on the
  second rank; `uni-cumps env` is reserved for external launchers
  ([#2499](https://github.com/Motphys/UniLab/pull/2499),
  [#2505](https://github.com/Motphys/UniLab/pull/2505)).
  多 GPU 拓扑保持显式且简单：省略 `CUDA_VISIBLE_DEVICES` 使用一个 rank；为每张
  选择的 GPU 提供一个条目则启用对应 rank。README 现在展示最小双卡 FlashSAC
  训练和手动管理的 CUDA MPS 流程。显式 `training.cuda_process_sharing=mps`
  请求会在 rank 派生和 CUDA 初始化前把数字 GPU 条目规范化为主机 UUID，因此
  文档中的 per-GPU MPS 启动不再在第二个 rank 失败；`uni-cumps env` 仅保留给
  外部启动器使用（[#2499](https://github.com/Motphys/UniLab/pull/2499)、
  [#2505](https://github.com/Motphys/UniLab/pull/2505)）。

### Fixed / 修复

- PPO, APPO, SAC, and FlashSAC entrypoints now handle `Ctrl+C` cooperatively:
  runner and collector cleanup finishes, the first interrupt returns shell
  status `130` without a traceback, and a second interrupt forces termination.
  Actual worker and rank failures remain loud
  ([#2503](https://github.com/Motphys/UniLab/pull/2503)).
  PPO、APPO、SAC 与 FlashSAC 入口现在协作处理 `Ctrl+C`：runner 与 collector
  清理完成后，第一次中断以 shell status `130` 退出且不输出 traceback；第二次
  中断强制终止。真实的 worker/rank 失败仍会完整暴露
  （[#2503](https://github.com/Motphys/UniLab/pull/2503)）。
- The off-policy live terminal display now starts at the collector-ready
  handshake, preventing learner compilation, CUDA graph capture, and collector
  startup messages from interleaving with live redraws
  ([#2507](https://github.com/Motphys/UniLab/pull/2507)).
  off-policy live 终端显示现在延迟到 collector-ready handshake，避免 learner
  编译、CUDA graph 捕获和 collector 启动消息与 live 刷新交错
  （[#2507](https://github.com/Motphys/UniLab/pull/2507)）。
- Off-policy runners initialize the learner inference timer on every supported
  device and record the effective `cuda_events` or synchronized wall-clock
  implementation in the runtime manifest. This fixes macOS MPS SAC/FlashSAC
  training failing at the first inference request
  ([#2509](https://github.com/Motphys/UniLab/pull/2509)).
  off-policy runner 会在每个受支持的 learner 设备上初始化推理计时器，并把实际
  使用的 `cuda_events` 或同步 wall-clock 实现记录到 runtime manifest。该修复解决
  macOS MPS SAC/FlashSAC 训练在第一次推理请求时失败的问题
  （[#2509](https://github.com/Motphys/UniLab/pull/2509)）。
- Fused community 13-D root-state and motion-state resets now rotate world-frame
  angular velocity into the generalized free-root/body frame before tensor
  submission, while linear velocity remains world-frame. Rotated roots with
  nonzero angular velocity therefore initialize the intended physical state
  without reintroducing host-boundary work
  ([#2513](https://github.com/Motphys/UniLab/pull/2513)).
  融合 community 13-D root-state 与 motion-state reset 现在会在 tensor 提交前把
  world-frame 角速度旋转到 generalized free-root/body frame，线速度仍保持
  world-frame。带非零角速度的旋转 root 因此能初始化为预期物理状态，且没有重新
  引入 host-boundary 开销（[#2513](https://github.com/Motphys/UniLab/pull/2513)）。

## 1.5.0 (2026-10-22)

### Breaking changes / 破坏性变更

- An explicit `training.collector_tensor_device=cpu` paired with a CUDA
  learner now fails closed on non-ROCm Torch builds: the CPU carrier is no
  longer a CUDA-learner performance path (issue #2479 retest showed throughput
  parity with CUDA carriers plus CUDA MPS). CPU carriers remain legal for CPU
  and torch.mps learners and for ROCm/HIP builds; attribution baselines and
  debug runs opt in via `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1`
  ([#2483](https://github.com/Motphys/UniLab/issues/2483), ADR-0019).
  显式 `training.collector_tensor_device=cpu` 与 CUDA learner 的组合在非 ROCm
  Torch 构建上现在 fail closed：CPU 载体不再是 CUDA learner 的性能路径
  （#2479 重测确认其与 CUDA 载体 + CUDA MPS 吞吐持平）。CPU 载体对 CPU /
  torch.mps learner 与 ROCm/HIP 构建仍然合法；归因基线与调试 run 通过
  `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1` 显式 opt-in（#2483，ADR-0019）。

- `training.sim_carrier_device` and `training.inference_transport` are removed
  without a compatibility layer; both merge into the single one-dimensional
  collector-tensor residency knob `training.collector_tensor_device`
  (`auto`/`cpu`/`cuda`, default `auto`, following the learner device). The
  off-policy inference transport is now a pure derived fact (`cpu` → host
  ring, `cuda` → CUDA ring). The env owner field is likewise renamed
  `env.sim_carrier_device` → `env.collector_tensor_device`. All removed keys
  fail closed with a migration hint; `run_config.json` records the mapping
  under `device_resolution.collector_tensor`
  ([#2481](https://github.com/Motphys/UniLab/issues/2481)).
  `training.sim_carrier_device` 与 `training.inference_transport` 已删除，
  无兼容层；两者合并为单一一维 collector tensor 驻留旋钮
  `training.collector_tensor_device`（`auto`/`cpu`/`cuda`，默认 `auto`，
  跟随 learner 设备）。off-policy inference transport 变为纯派生事实
  （`cpu` → host ring，`cuda` → CUDA ring）。env owner 字段同步改名
  `env.sim_carrier_device` → `env.collector_tensor_device`。全部已移除键
  带迁移提示 fail closed；`run_config.json` 在
  `device_resolution.collector_tensor` 记录映射（#2481）。

- The public reset contract speaks IsaacLab/mjlab-style `env_ids` row indices
  again: `TorchEnv.reset`, `ManagerBase.reset` and all manager/term reset
  boundaries, `EventManager.apply(mode="reset", env_ids=...)`, reset owners,
  and the recorder pre/post-reset hooks take a one-dimensional integer tensor
  (or `None` for every row). The v1.4 fixed-width bool mask form is removed
  without a compatibility layer: a bool tensor passed positionally now raises
  `TypeError`, mask-form reset event terms (`reset(env, reset_mask, **params)`)
  are no longer detected or dispatched, and the `EventManager.dispatch_reset`
  wrapper is gone — use `apply(mode="reset", env_ids=...)`. The keyword form
  likewise raises `TypeError`; rename the argument to `env_ids` and pass row
  indices
  ([#2460](https://github.com/Motphys/UniLab/issues/2460),
  [#2474](https://github.com/Motphys/UniLab/issues/2474)).
  公共 reset 契约恢复 IsaacLab/mjlab 风格的 `env_ids` 行索引形态：
  `TorchEnv.reset`、`ManagerBase.reset` 及所有 manager/term reset 边界、
  `EventManager.apply(mode="reset", env_ids=...)`、reset owner 与 recorder
  的 pre/post-reset 钩子均接收一维整数张量（`None` 表示全部行）。v1.4 的
  定宽 bool mask 形态已直接移除、无兼容层：位置传参的 bool 张量现在抛
  `TypeError`；mask 形态的 reset 事件项（`reset(env, reset_mask, **params)`）
  不再被检测或派发；`EventManager.dispatch_reset` 包装已删除——请使用
  `apply(mode="reset", env_ids=...)`。关键字形式同样抛 `TypeError`；请将
  参数名改为 `env_ids` 并传行索引（#2460，#2474）。
  迁移步骤与逐类 term 的 before/after 示例见迁移指南
  {doc}`/en/1-getting_started/6-migration_1_4_to_1_5` /
  {doc}`/zh_CN/1-getting_started/6-migration_1_4_to_1_5`。

### Added / 新增

- `training.tf32_matmul` is a new PPO/APPO float32 matmul backend knob:
  `ieee` (default) preserves full-precision FP32, and `tf32` opts into TF32
  tensor-core kernels with an automatic IEEE fallback on MPS/macOS, ROCm/HIP,
  pre-Ampere GPUs (compute capability below 7.0), and CPU. Training
  entrypoints apply the policy before CUDA and learner construction, play
  paths honor it, and the resolved value plus fallback reason are recorded in
  `run_config.json`/`run_summary.json`. The knob is intentionally not exposed
  for SAC-family learners: their bf16-AMP updates already cast the main
  matmuls, and the closed #2477 benchmark measured TF32 as neutral there.
  The main benefit is the FP32 PPO/RSL-RL update path
  ([#2475](https://github.com/Motphys/UniLab/issues/2475)).
  新增 PPO/APPO 的 float32 matmul 后端开关 `training.tf32_matmul`：`ieee`
  （默认）保持全精度 FP32；`tf32` 启用 TF32 tensor-core kernel，并在
  MPS/macOS、ROCm/HIP、pre-Ampere GPU（compute capability < 7.0）与 CPU 上自动
  回退为 IEEE。训练入口会在 CUDA 与 learner 构造前应用该策略，play 路径同样
  生效，解析结果与回退原因写入 `run_config.json`/`run_summary.json`。该键有意
  不暴露给 SAC 家族：其 bf16-AMP 更新已接管主要 matmul，已关闭的 #2477 实测
  TF32 在该路径为噪声级。主要收益在 FP32 的 PPO/RSL-RL 更新路径（#2475）。

- Manager-owned observation history and delay are restored:
  `ObservationTermCfg.history_length > 0` and `delay_*` construct and reset
  again, with the v1.3.4 semantics (oldest-to-newest flattening, first-frame
  and post-reset backfill, noise/clip/scale before history). Partial resets
  backfill the reset rows and peek without advancing other rows' history,
  delay schedule, or lag RNG. `CircularBuffer` and `DelayBuffer` are now
  exported publicly from `unilab.managers` so downstream temporal owners no
  longer import the private `unilab.managers._buffers` package
  ([#2462](https://github.com/Motphys/UniLab/issues/2462)).
  恢复 Manager 持有的观测 history/delay：`ObservationTermCfg.history_length > 0`
  与 `delay_*` 重新可构造、可 reset，语义与 v1.3.4 一致（oldest→newest 展开、
  首帧与 reset 后 backfill、noise/clip/scale 先于 history）。部分 reset 只对
  reset 行做 backfill 并 peek，不推进其他行的 history、delay 日程或 lag RNG。
  `CircularBuffer` 与 `DelayBuffer` 现从 `unilab.managers` 公开导出，下游
  temporal owner 不再需要 import 私有包 `unilab.managers._buffers`（#2462）。

### Changed / 变更

- The internal `env_ids` → bool mask adaptation layer now scatters into
  owner-held reusable buffers (`unilab.base.reset_selectors.ResetMaskBuffer`)
  instead of allocating a fresh `num_envs` bool tensor on every reset/event
  term call: the RL env reset entry, the event/observation/command managers,
  and per-term reset owners each hold their own scratch buffer, so the hot
  reset path no longer pays a per-call allocation. Public contracts are
  unchanged; `reset_env_ids_mask` keeps its allocating form for fixtures and
  external callers
  ([#2461](https://github.com/Motphys/UniLab/issues/2461)).
  内部 `env_ids` → bool mask 适配层改为 scatter 进 owner 持有的可复用
  buffer（`unilab.base.reset_selectors.ResetMaskBuffer`），不再在每次
  reset/event term 调用时新分配 `num_envs` 大小的 bool 张量：RL env reset
  入口、event/observation/command manager 及各 term 的 reset owner 分别持有
  各自的 scratch buffer，热路径上消除了逐次分配。公开契约不变；
  `reset_env_ids_mask` 保留逐次分配形态供 fixture 与外部调用方使用（#2461）。

### Fixed / 修复

- Fixed the mjwarp off-policy collector `Env Step` breakdown hiding
  `Backend Step`: an event-backed asynchronous step's physics wait surfaces at
  the first host read of the step outputs — the update-state termination drain,
  or the autoreset done-mask gate when update_state performs no host read — and
  was misattributed to `Update State` / `Reset Done`. Both windows now roll up
  into `Backend Step` through `TorchEnv._collect_step_core_drain_ms`; total
  `Env Step` accounting is unchanged and synchronous backends keep their
  existing attribution. The off-policy terminal System column also gained a
  `CUDA MPS` indicator (`on` / `ambient` / `off`) driven by the run's
  `cuda_process_sharing` manifest evidence
  ([#2495](https://github.com/Motphys/UniLab/issues/2495)).
  修复 mjwarp off-policy collector `Env Step` 分解缺 `Backend Step` 的问题：
  事件型异步 step 的物理等待发生在 step 输出后的第一个 host 读取处
  （update_state 内 termination drain；若 update_state 无 host 读取则为
  autoreset done-mask 门控），此前被误归入 `Update State` / `Reset Done`。
  两个窗口现在经 `TorchEnv._collect_step_core_drain_ms` 归入 `Backend Step`；
  `Env Step` 总额不变，同步后端归因不变。off-policy 终端 System 列新增
  `CUDA MPS` 指示（`on` / `ambient` / `off`），取值自 run 的
  `cuda_process_sharing` manifest 证据（#2495）。

## 1.4.0 (2026-10-20)

### Breaking changes / 破坏性变更

- Resuming PPO/APPO training now fails closed on pre-patch rsl-rl checkpoints:
  a checkpoint without a valid `unilab_logger_state` — and, for environments
  that publish training state, without a valid versioned
  `unilab_env_training_state` — is rejected with an error naming the minimum
  supported format instead of silently restarting progress from 0. There is no
  compatibility window, offline migration tool, or dual-write path; retrain or
  resume from a checkpoint saved by a supported UniLab version
  ([#2416](https://github.com/Motphys/UniLab/issues/2416)).
  PPO/APPO 续训对 pre-patch rsl-rl checkpoint fail closed：缺少有效
  `unilab_logger_state` 的 checkpoint——对于发布 training state 的环境，还需有效的
  版本化 `unilab_env_training_state`——会被拒绝加载，错误信息说明最低支持格式，
  不再静默地从 0 恢复进度。无兼容窗口、离线迁移工具或 dual-write 路径；请用受支持
  的 UniLab 版本重训，或从该版本保存的 checkpoint 续训（#2416）。

- The sim2sim contract guard no longer warning-skips historical runs: a source
  run whose `run_config.json` is missing, malformed, lacks a mapping
  `contract_snapshot`, or contains keys outside the current contract schema
  fails closed with `CrossBackendIncompatibleError`. The legacy snapshot/config
  key aliases (`env.sampling_mode`, `algo.policy.actor_hidden_dims`,
  `algo.policy.critic_hidden_dims`, `algo.empirical_normalization`) are removed;
  only canonical `contract_snapshot` mappings written by training runs on the
  1.3.5 baseline schema are supported, and this change is not backported to
  1.3.5 patch releases. To reuse an older run, retrain with the current version
  (checkpoint weights are unchanged); `training.sim2sim_strict=false` does not
  bypass the integrity failures
  ([#2416](https://github.com/Motphys/UniLab/issues/2416)).
  Sim2Sim 契约守卫不再对历史 run warning-skip：源 run 的 `run_config.json`
  缺失、损坏、缺少 mapping 形态的 `contract_snapshot`、或含有当前契约 schema
  之外的键时一律 fail closed（抛 `CrossBackendIncompatibleError`）。legacy
  快照/配置键别名（`env.sampling_mode`、`algo.policy.actor_hidden_dims`、
  `algo.policy.critic_hidden_dims`、`algo.empirical_normalization`）已移除；
  仅支持 1.3.5 基线 schema 训练 run 写出的 canonical `contract_snapshot`，
  且本变更不回 port 到 1.3.5 patch release。要复用旧 run 需用当前版本重训
  （checkpoint 权重本身不变）；`training.sim2sim_strict=false` 不能绕过完整性
  失败（#2416）。
- FlashSAC pre-contract checkpoints (a missing or null `actor_semantics`
  entry) are no longer supported: training resume and play loading fail
  closed for every actor, including the built-in actor, with an error that
  states the minimum supported checkpoint format. The only migration path is
  to retrain the policy with a current release; the change is not backported
  to any 1.3.5 patch release
  ([#2416](https://github.com/Motphys/UniLab/issues/2416)).
  FlashSAC pre-contract checkpoint（缺失或为 null 的 `actor_semantics` 条目）
  不再受支持：训练续训与 play 加载对所有 actor（包括内置 actor）一律 fail
  closed，错误信息说明最低支持的 checkpoint 格式。唯一迁移路径是使用当前
  版本重新训练策略；该变更不回 port 到任何 1.3.5 patch release（#2416）。
- The training seed contract no longer falls back to `training.seed`; only the
  algorithm-level `algo.seed` is honored. A config that sets `training.seed`
  without `algo.seed` fails closed with an error pointing to `algo.seed`
  ([#2415](https://github.com/Motphys/UniLab/issues/2415)).
  训练 seed 契约不再 fallback 到 `training.seed`，仅认可算法级
  `algo.seed`。配置了 `training.seed` 但缺少 `algo.seed` 的配置会 fail
  closed，错误信息指向 `algo.seed`（#2415）。
- `render_play_mode()` no longer accepts the ignored `sim_backend` keyword;
  backend routing belongs to `env.run_playback()`. Callers that still pass the
  argument now get a `TypeError`; deleting the argument migrates the call with
  unchanged behavior
  ([#2415](https://github.com/Motphys/UniLab/issues/2415)).
  `render_play_mode()` 不再接受无效的 `sim_backend` 关键字参数；backend 路由
  归属 `env.run_playback()`。仍传该参数的调用方会得到 `TypeError`；删除该
  实参即可完成迁移，行为不变（#2415）。
- Interactive playback no longer supports the non-Hydra `cfg=None` path:
  `create_playback_session()` requires a composed Hydra `DictConfig`, and the
  sim2sim preflight fails closed instead of being silently skipped when the
  config is missing or is not a `DictConfig`
  ([#2415](https://github.com/Motphys/UniLab/issues/2415)).
  交互式回放不再支持非 Hydra 的 `cfg=None` 路径：`create_playback_session()`
  要求组合后的 Hydra `DictConfig`；config 缺失或不是 `DictConfig` 时
  sim2sim preflight fail closed，不再静默跳过（#2415）。
- The Isaac setup scripts accept only `UNISIM_ISAACGYM_HOME` /
  `UNISIM_ISAACSIM_HOME`; the retired `UNILAB_ISAACGYM_HOME` /
  `UNILAB_ISAACSIM_HOME` spellings fail closed with an error naming the current
  variables ([#2415](https://github.com/Motphys/UniLab/issues/2415)).
  Isaac 安装脚本仅接受 `UNISIM_ISAACGYM_HOME` / `UNISIM_ISAACSIM_HOME`；已
  退役的 `UNILAB_ISAACGYM_HOME` / `UNILAB_ISAACSIM_HOME` 拼写会 fail
  closed，错误信息指向当前变量名（#2415）。
- `unilab-pull-assets` no longer accepts the no-op `--verbose` option; passing
  it now fails argument parsing. Output stays compact by default
  ([#2415](https://github.com/Motphys/UniLab/issues/2415)).
  `unilab-pull-assets` 删除 no-op 的 `--verbose` 选项；传入该选项会导致参数
  解析失败。输出默认保持紧凑（#2415）。
- Benchmark tooling drops its legacy compatibility surface: physics benchmarks
  accept only the canonical owner task ids (`go2_joystick_flat`,
  `g1_walk_flat`; env-class names, route forms, and short names are rejected),
  replay placement accepts only `--tasks` (the singular `--task` alias is
  removed), the SAC replay-sampling benchmark's legacy defaults and positional
  sim argument are removed, the IsaacGym/IsaacSim benchmark scripts drop the
  `UNILAB_ISAACGYM_DEPS_ROOT` / `UNILAB_ISAACGYM_MODELS_ROOT` environment
  aliases and machine-local path defaults (unset paths fail closed), and the
  Torch benchmark workloads publish a single `final_observation` carrier
  instead of the dual-written `compat_final_obs` buffer
  ([#2413](https://github.com/Motphys/UniLab/issues/2413)).
  benchmark 工具链删除 legacy 兼容面：physics benchmark 仅接受 canonical
  owner task id（`go2_joystick_flat`、`g1_walk_flat`；env-class 名、路由形式
  与短名一律拒绝）；replay placement 仅接受 `--tasks`（删除单数 `--task`
  别名）；SAC replay sampling benchmark 删除旧默认值与位置参数 sim 解释；
  IsaacGym/IsaacSim benchmark 脚本删除 `UNILAB_ISAACGYM_DEPS_ROOT` /
  `UNILAB_ISAACGYM_MODELS_ROOT` 环境变量别名与机器本地路径默认值（未设置时
  fail closed）；Torch benchmark 工作负载只发布单一 `final_observation`
  载体，不再双写 `compat_final_obs` 缓冲（#2413）。
- `training.collector_tensor_device`, `manager_torch_device`, `env.tensor_runtime`,
  and `env.tensor_runtime_device` are removed. Use `training.sim_carrier_device`
  (routed to `env.sim_carrier_device`) and the resolved learner device for
  tensor-native inference. Removed keys fail closed with migration hints, `cuda`
  no longer maps silently to MPS, and `run_config.json` records configured-to-resolved
  devices. The required `unisim-core` 1.7.25 enum support inventory and runtime
  parity guard are pinned ([#2382](https://github.com/Motphys/UniLab/issues/2382)).
  `training.collector_tensor_device`、`manager_torch_device`、`env.tensor_runtime`
  与 `env.tensor_runtime_device` 已移除。请使用
  `training.sim_carrier_device`（路由到 `env.sim_carrier_device`），并让 tensor-native
  推理跟随已解析 learner 设备。移除键会带迁移提示 fail closed，`cuda` 不再静默映射
  MPS，`run_config.json` 会记录“配置值 → 解析值”设备映射。所需
  `unisim-core` 1.7.25 枚举支持清单与 runtime parity 守卫已固定。
- The former `unilab-rl` distribution (import namespace `uni_rl`) is merged
  into the `unilab` package under `unilab.rl` (contracts / adapters / algos /
  runtime / logging layers), and the release surface converges to the single
  `unilab` distribution. The transitional `uni_rl` extra has been removed:
  legacy `--extra uni_rl` / `pip install unilab[uni_rl]` commands are no longer
  valid, and permanent `import uni_rl` support is not guaranteed. See
  {doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl` and the
  `uni_rl.*` → `unilab.rl.*` import-path mapping in
  {doc}`/en/3-deployment/3-framework_migration/7-from_unilab_rl`
  ([#2372](https://github.com/Motphys/UniLab/issues/2372)).
  原 `unilab-rl` distribution（import namespace `uni_rl`）并入 `unilab` 包的
  `unilab.rl`（contracts / adapters / algos / runtime / logging 分层），
  发行面收敛为单一 `unilab` distribution。过渡期 `uni_rl` extra 已移除，
  旧 `--extra uni_rl` / `pip install unilab[uni_rl]` 命令不再有效，且不承诺
  永久支持 `import uni_rl`。决策见 ADR-0016，`uni_rl.*` → `unilab.rl.*` 的 import
  路径映射见迁移页
  {doc}`/zh_CN/3-deployment/3-framework_migration/7-from_unilab_rl`（#2372）。
- Removed the collector CUDA-graph public lifecycle and graph-named completion
  contract without aliases or compatibility paths. Runtime code now consumes the
  neutral completion-v2 receipt (`state.info["completion"]`) while eager ordered
  visibility, producer validation, actual-operation waits, backend-owned physics
  graphs, and learner CUDA graphs remain in their owner layers. Each successful
  step publishes its current receipt, and an autoreset step publishes the reset
  boundary receipt that gates its post-reset observation carrier
  ([#2359](https://github.com/Motphys/UniLab/issues/2359),
  [#2358](https://github.com/Motphys/UniLab/issues/2358)).
  移除 collector CUDA graph 的公共 lifecycle 与 graph 命名 completion 契约，
  不保留 alias 或兼容路径。运行时改用中性 completion-v2 receipt
  （`state.info["completion"]`），同时保留 eager 有序可见性、producer 校验、
  按实际操作等待、backend 私有 physics graph 与 learner CUDA graph
  （#2359、#2358）。

### Changed / 变更

- Raised the required `unisim-core` pin to 1.8.2 for the base, Motrix, and
  SuperDex dependency profiles, and removed the issue-local editable UniSim
  source override. The published release supplies the restored packed-reset,
  Isaac semantic-boundary, and Motrix `wxyz` FrameQuat contracts.
  基础、Motrix 与 SuperDex 依赖配置将 `unisim-core` 提升到 1.8.2，并移除
  issue 本地的 editable UniSim source override。已发布版本提供恢复后的
  packed-reset、Isaac 语义边界与 Motrix `wxyz` FrameQuat 契约。
- `training.collector_tensor_device` now defaults to `auto`: MuJoCo
  HOST_BRIDGE training follows the learner device, so a CUDA learner receives
  CUDA Manager/TorchEnv carriers and CUDA inference transport without explicit
  overrides, while CPU and torch.mps (Metal Performance Shaders) learners keep
  the CPU carrier. Explicit `cpu`/`cuda` values are unchanged and win over
  `auto`. Existing MuJoCo configurations that relied on the previous `cpu`
  default with a CUDA learner now allocate one CUDA context per collector
  process on the learner GPU; pin `training.collector_tensor_device=cpu` to
  keep the previous topology
  ([#2365](https://github.com/Motphys/UniLab/issues/2365)).
  `training.collector_tensor_device` 默认值改为 `auto`：MuJoCo HOST_BRIDGE
  训练跟随 learner 设备——CUDA learner 无需显式配置即可获得 CUDA
  Manager/TorchEnv carrier 与 CUDA inference transport，CPU 与 torch.mps
  （Metal Performance Shaders）learner 保持 CPU carrier。显式 `cpu`/`cuda`
  语义不变且优先于 `auto`。原先在 CUDA learner 下依赖 `cpu` 默认值的
  MuJoCo 配置，现在每个 collector 进程会在 learner GPU 上额外占用一个
  CUDA context；如需保持原拓扑，请显式固定
  `training.collector_tensor_device=cpu`（#2365）。

### Fixed / 修复

- Fixed capability-scoped Manager carrier routing for every in-process packed
  `HOST_BRIDGE` backend, not only MuJoCo. Explicit
  `training.sim_carrier_device=cuda` and the default `auto` routing now honor
  the public backend `torch_devices` declaration for Motrix, Drake, and
  SuperDex, while unsupported or unknown device declarations fail closed.
  Physics remains CPU-authoritative; this changes only public Manager carrier
  placement ([#2401](https://github.com/Motphys/UniLab/issues/2401)).
  修复 Manager carrier 路由仅限 MuJoCo 的问题：显式
  `training.sim_carrier_device=cuda` 与默认 `auto` 现在按公开的后端
  `torch_devices` 声明作用于 Motrix、Drake 与 SuperDex 等全部 in-process
  packed `HOST_BRIDGE` 后端；未声明或不受支持的设备会 fail closed。物理仍
  以 CPU 为权威，本修复只改变公共 Manager carrier 放置位置。
- Fixed device-resident Manager sensor staleness and restored SAC
  `g1_walk_flat` policy quality across the newly adapted backends. Every
  compiled derived sensor now crosses its backend boundary on full and
  selected-reset reads; IsaacSim uses a task-wide finite floor with audited
  50 m clone spacing; and IsaacGym clamps position targets to source MJCF joint
  bounds. Full 5000-iteration SAC runs recover Genesis/Newton/IsaacSim/IsaacGym
  rewards and episode lengths comparable with the MuJoCo-family baselines
  ([#2387](https://github.com/Motphys/UniLab/issues/2387)).
  修复 device-resident Manager sensor 停滞问题，并恢复新适配后端在 SAC
  `g1_walk_flat` 上的策略质量。每个 compiled derived sensor 在 full 与
  selected-reset 读取时都会重新跨越 backend 边界；IsaacSim 使用覆盖任务的
  finite floor 并审计 50 m clone 间距；IsaacGym 将 position target clamp 到
  source MJCF joint bound。四条 5000-iteration SAC 完整运行的 reward 与
  episode length 已回到 MuJoCo-family baseline 区间（#2387）。
- Off-policy replay timeout bootstrapping: a transition that returns
  `terminated=True` and `truncated=True` together is a true terminal and no
  longer bootstraps. The collector now writes the timeout-bootstrap
  eligibility mask (`truncated & ~terminated`) into the replay `truncated`
  column while `dones` keeps the `terminated | truncated` lifecycle semantics
  and done rows keep the real terminal observations; SAC/FlashSAC targets are
  unchanged in form and now correct for the overlap case
  ([unilab_rl#29](https://github.com/unilabsim/unilab_rl/issues/29)).
  修复 off-policy replay 的 timeout bootstrap：同一 transition 同时返回
  `terminated=True` 与 `truncated=True` 时是真正的终止，不再 bootstrap。
  collector 现在把 timeout bootstrap 资格掩码（`truncated & ~terminated`）
  写入 replay 的 `truncated` 列；`dones` 保持 `terminated | truncated`
  的生命周期语义，done 行仍携带真实 terminal observation；SAC/FlashSAC 的
  target 公式形式不变，overlap 情形结果修正（unilab_rl#29）。
- Multi-term concatenated observations are again caller-owned: the Manager no
  longer publishes rotating internal output carriers through the default eager
  boundary, so observations retained by a transition consumer remain immutable
  across later steps ([#2362](https://github.com/Motphys/UniLab/issues/2362)).
  多 term 拼接 observation 重新归 caller 所有：Manager 不再通过默认 eager
  boundary 发布轮换的内部 output carrier，因此 transition consumer 保留的
  observation 不会在后续 step 中被改写（#2362）。
- Restored the v1.3.5 `MotionCommand` empty-reset-mask semantics: an empty mask
  consumes no sampler RNG and performs no backend state write or packet ingestion
  ([#2359](https://github.com/Motphys/UniLab/issues/2359)).
  恢复 v1.3.5 的 `MotionCommand` 空 reset mask 语义：空 mask 不消耗 sampler
  RNG，也不写 backend state 或摄取 packet（#2359）。
- Fixed IsaacGym/IsaacSim interactive eval, which failed before the native
  viewer opened: unisim-core 2.0.0 fails closed when an Isaac native
  interactive viewer receives a non-default `camera_kwargs` profile, while
  the eval play path forwards `training.cam_*` to every backend.
  `run_playback_mode` now mirrors the interactive debug-overlay degradation —
  it warns, drops the camera profile, and retries interactive playback;
  record mode still forwards the profile and fails closed on rejection
  ([#2454](https://github.com/Motphys/UniLab/issues/2454)).
  修复 IsaacGym/IsaacSim 交互 eval 在原生 viewer 打开前失败的问题：
  unisim-core 2.0.0 对收到非默认 `camera_kwargs` 的 Isaac 原生交互 viewer
  fail closed，而 eval 播放路径会把 `training.cam_*` 传给所有后端。
  `run_playback_mode` 现在与 interactive debug-overlay 的降级模式一致——
  告警、去掉相机配置并重试交互播放；record 模式仍照常下发相机配置，
  被拒绝时 fail closed（#2454）。

## 1.3.5 (2026-10-13)

### Breaking changes / 破坏性变更

- Completed the production Torch-runtime closeout: UniLab no longer imports,
  compiles, tunes, or requires Numba. The CPU motion command and kernels were
  removed, `cpu_ids` owns only Linux process affinity, and every G1 motion
  owner uses the device-resident `MotionCommandCfg`. Manager, env, and task
  runtime code uses Torch-authoritative carriers and rejects hidden
  Torch↔NumPy conversions; the cold NPZ `MotionLoader` is the only task-runtime
  NumPy leaf, while base owners confine legacy NumPy exchange to explicit
  adapter/reset-host boundaries
  ([#2124](https://github.com/Motphys/UniLab/issues/2124),
  [ADR-0015](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md)).
  Production Numba 实现与依赖已完整移除：UniLab 不再 import、编译、调优或
  依赖 Numba。CPU motion command 与 kernels 已删除，`cpu_ids` 仅保留 Linux
  process affinity 语义，所有 G1 motion owner 均使用 device-resident
  `MotionCommandCfg`。Manager、env 与 task runtime 使用 Torch-authoritative
  carrier，并拒绝隐藏的 Torch↔NumPy 转换；cold NPZ `MotionLoader` 是
  task runtime 唯一 NumPy leaf，base owner 只在显式 adapter/reset-host
  boundary 内消化 legacy NumPy 交换（#2124、ADR-0015）。

- The Manager-Based reset lifecycle now fails closed unless the backend declares
  `packed_reset`. UniLab's NumPy-authoritative reset composer and non-packed
  fallback were removed: reset-mode state, model-field, and mocap writes are
  composed through UniSim's public packed transaction and committed once per
  lifecycle. UniLab no longer calls `SimBackend.set_state_tensor_mask()`;
  mapped-entity tensor commits remain an explicit migration boundary
  ([#2275](https://github.com/Motphys/UniLab/issues/2275),
  [#2280](https://github.com/Motphys/UniLab/issues/2280),
  [#2289](https://github.com/Motphys/UniLab/pull/2289)).
  Manager-Based reset lifecycle 现在要求 backend 声明 `packed_reset`，否则
  fail closed。UniLab 侧 NumPy-authoritative reset composer 与 non-packed
  fallback 已删除：reset-mode 状态、模型字段与 mocap 写入通过 UniSim 公共
  packed transaction 组合，并在每个生命周期恰好提交一次。UniLab 不再调用
  `SimBackend.set_state_tensor_mask()`；mapped-entity tensor 提交仍是显式
  迁移边界（#2275、#2280、#2289）。

- The training seed contract moved from the `unilab.utils.seed` transition shim
  to its owner at `unilab.training.seed`, without a compatibility alias.
  `unilab.training` continues to re-export the owner-facing helpers.
  `algo.seed` now seeds only global Python and Torch RNGs, and random
  interactive playback samples actions with Torch
  ([#2124](https://github.com/Motphys/UniLab/issues/2124),
  [#2283](https://github.com/Motphys/UniLab/issues/2283)).
  训练 seed 契约已从 `unilab.utils.seed` 过渡 shim 迁移到 owner
  `unilab.training.seed`，且不保留兼容 alias。`unilab.training` 继续导出
  owner-facing helper。`algo.seed` 现在只 seed 全局 Python 与 Torch RNG，
  随机 interactive playback 改用 Torch 采样 action（#2124、#2283）。

- Removed the `unilab.utils.rotation` transition shim without a compatibility
  alias. Base-owner host math moved to `unilab.base.rotation`, while motion NPZ
  continuity and angular-velocity conversion now live inside the cold
  `MotionLoader` owner.
  过渡 shim `unilab.utils.rotation` 已移除且不保留兼容 alias。base-owner
  host 数学迁移到 `unilab.base.rotation`；motion NPZ 连续性与角速度转换则
  收进 cold `MotionLoader` owner 内部。

### Added / 新增

- Added backend tensor-runtime diagnostics. `TorchEnv` serializes the
  public backend diagnostics, training records them in
  `runtime_manifest.backend_diagnostics`, and direct PPO runs publish the same
  payload as top-level `backend_diagnostics` in `run_summary.json`; unavailable
  backends and probe failures are recorded explicitly
  ([#2278](https://github.com/Motphys/UniLab/issues/2278),
  [#2298](https://github.com/Motphys/UniLab/pull/2298)).
  新增 backend tensor-runtime diagnostics。`TorchEnv` 会序列化公共
  backend diagnostics，训练将其记录到
  `runtime_manifest.backend_diagnostics`，direct PPO run 会在
  `run_summary.json` 顶层发布同一 payload；backend 不支持或探测失败都会显式
  记录（#2278、#2298）。

### Changed / 变更

- Manager terms and tasks consume semantic, packed tensor state views for
  joint, body, root, contact, height, and sensor data. Legacy NumPy backend
  methods are no longer UniLab runtime APIs, and Entity host exchange is
  confined to the backend adapter boundary
  ([#2276](https://github.com/Motphys/UniLab/issues/2276),
  [#2277](https://github.com/Motphys/UniLab/issues/2277)).
  Manager term 与 task 改为消费 joint、body、root、contact、height 与
  sensor 数据的 semantic packed tensor view。legacy NumPy backend 方法不再是
  UniLab runtime API，Entity host 交换收进 backend adapter 边界
  （#2276、#2277）。

- Selected reset execution replaced dynamically materialized row carriers with
  fixed-width packed masks. Commands, events, metrics, observations,
  terminations, curricula, and task state implement explicit mask-reset
  boundaries, commit only touched rows, and preserve active state elsewhere
  ([#2285](https://github.com/Motphys/UniLab/issues/2285),
  [#2288](https://github.com/Motphys/UniLab/issues/2288)).
  selected reset 执行以 fixed-width packed mask 取代动态物化 row carrier。
  command、event、metric、observation、termination、curriculum 与 task state
  均实现显式 mask-reset 边界，只提交被写入的行，并保留其他行的 active 状态
  （#2285、#2288）。

- UniSim now owns packed reset composition: selected-row device qpos/qvel
  staging, selected-column merges, committed model-field baselines, startup
  baseline promotion, selected mocap staging, and the single public commit
  boundary. The composer is cached per declared device and its baseline
  persists across resets ([#2275](https://github.com/Motphys/UniLab/issues/2275)).
  UniSim 现在拥有 packed reset composition：selected-row device qpos/qvel
  staging、选中列合并、已提交模型字段 baseline、startup baseline 提升、
  selected mocap staging，以及唯一公共提交边界。composer 按声明 device
  缓存，baseline 跨 reset 持久（#2275）。

- Required runtime dependencies moved to `unisim-core>=1.7.17` and
  `unilab-rl==1.4.12`; the default and ROCm dependency profiles are aligned
  ([#2281](https://github.com/Motphys/UniLab/issues/2281),
  [#2299](https://github.com/Motphys/UniLab/pull/2299),
  [#2301](https://github.com/Motphys/UniLab/pull/2301)).
  必需运行时依赖升级为 `unisim-core>=1.7.17` 与 `unilab-rl==1.4.12`；
  default 与 ROCm dependency profile 已对齐（#2281、#2299、#2301）。

### Fixed / 修复

- Fixed MuJoCo reset model-field DR commits through the host bridge, including
  startup randomization baselines
  ([#2122](https://github.com/Motphys/UniLab/issues/2122),
  [#2123](https://github.com/Motphys/UniLab/pull/2123)).
  修复 MuJoCo reset 模型字段 DR 经 host bridge 的提交，包括 startup
  randomization baseline（#2122、#2123）。

- Fixed canonical off-policy packed-reset paths: root and motion payloads are
  normalized to contiguous float32 tensors, owner-subset commands track their
  actual fixed-width write mask while preserving unwritten active rows, and SAC
  MuJoCo PD gains compose as final kp/kd model fields instead of silently
  skipping randomization
  ([#2296](https://github.com/Motphys/UniLab/issues/2296),
  [#2300](https://github.com/Motphys/UniLab/pull/2300)).
  修复典型 off-policy packed-reset 路径：root 与 motion payload 归一化为
  contiguous float32 tensor，owner-subset command 记录实际 fixed-width 写入
  mask 并保留未写入的 active rows，SAC MuJoCo PD gain 以最终 kp/kd 模型字段
  组合，而不是静默跳过随机化（#2296、#2300）。

- Fixed motion NPZ body-column selection by local IDs, CUDA Manager carrier
  placement, and G1 reset-row observation boundaries
  ([#2124](https://github.com/Motphys/UniLab/issues/2124),
  [#2279](https://github.com/Motphys/UniLab/issues/2279),
  [#2303](https://github.com/Motphys/UniLab/issues/2303)).
  修复按 local ID 选择 motion NPZ body column、CUDA Manager carrier placement，
  以及 G1 reset-row observation 边界（#2124、#2279、#2303）。

## 1.3.4 (2026-10-10)

### Breaking changes / 破坏性变更

- The Manager-Based runtime is now tensor-only: `NpEnv` and `NpEnvState` were
  removed, `TorchEnv` is the sole environment lifecycle base, and Manager state,
  observations, rewards, terminations, commands, events, metrics, and recorder
  carriers use Torch tensors. Hot-path step/reset inputs no longer accept
  NumPy arrays or perform hidden host conversion
  ([ADR-0011](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0011-torch-only-manager-based-runtime.md),
  [ADR-0012](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0012-sole-tensor-manager-and-scoped-backends.md)).
  Manager-Based 运行时改为 tensor-only：删除 `NpEnv` / `NpEnvState`，
  `TorchEnv` 成为唯一环境生命周期基类；Manager 的状态、观测、奖励、终止、
  command、event、metric 与 recorder carrier 均使用 Torch tensor。热路径
  step/reset 不再接受 NumPy 输入，也不再做隐藏 host 转换（ADR-0011、
  ADR-0012）。

- Removed the production Numba implementation and dependency: the CPU motion
  command and its kernels were removed, `cpu_ids` now owns only Linux process
  affinity, and all G1 motion owners use the device-resident
  `MotionCommandCfg`. Manager/env runtime packages reject NumPy imports and
  hidden Torch↔NumPy conversions; the cold NPZ MotionLoader remains the only
  task-runtime NumPy leaf, while base owners consume legacy NumPy formats only
  in explicit adapter/reset-host boundaries
  ([#2124](https://github.com/Motphys/UniLab/issues/2124),
  [ADR-0015](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md)).
  移除 production Numba 实现与依赖：删除 CPU motion command 和 kernels，
  `cpu_ids` 仅保留 Linux process affinity 语义，所有 G1 motion owner 均使用
  device-resident `MotionCommandCfg`。Manager/env runtime package 禁止 NumPy
  import 与隐藏 Torch↔NumPy 转换；cold NPZ MotionLoader 仍是 task runtime
  唯一 NumPy leaf，base owner 只在显式 adapter/reset-host boundary 内消化
  legacy NumPy format（#2124、ADR-0015）。

- The training-entry seed contract no longer mutates the legacy global NumPy RNG;
  `algo.seed` seeds global Python and Torch RNGs only. Random interactive
  playback also samples its actions with Torch instead of `np.random.uniform`.
  This removes the last environment-owned global NumPy RNG semantics from
  training (#2124).
  训练入口 seed 契约不再修改 legacy 全局 NumPy RNG；`algo.seed` 只 seed
  全局 Python 与 Torch RNG。随机 interactive playback 也改用 Torch 采样
  action，不再调用 `np.random.uniform`。该变更移除训练中最后一个
  environment-owned 全局 NumPy RNG 语义（#2124）。

- The transition-shim module `unilab.utils.rotation` was removed without a
  compatibility alias. Its base-owner host math moved to
  `unilab.base.rotation`, while motion NPZ continuity and angular-velocity
  conversion now live inside the cold `MotionLoader` owner.
  过渡 shim `unilab.utils.rotation` 已移除且不保留兼容 alias。base-owner
  host 数学迁移到 `unilab.base.rotation`；motion NPZ 连续性与角速度转换则
  收进 cold `MotionLoader` owner 内部。

- Production training now enters through the sole Manager-Based factory and the
  scoped tensor backend set. Retired task owners, the WarpSAC config tree, and
  task-owned direct training environment classes were removed; fused task
  components remain Manager-owned terms rather than alternate environments
  ([#2071](https://github.com/Motphys/UniLab/pull/2071),
  [#2072](https://github.com/Motphys/UniLab/pull/2072),
  [ADR-0012](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0012-sole-tensor-manager-and-scoped-backends.md)).
  生产训练统一进入唯一的 Manager-Based factory 与 scoped tensor backend 集。
  退役 task owner、WarpSAC 配置树和 task-owned direct environment class 均已
  移除；融合任务组件保持为 Manager-owned term，而不是另一套环境类。

- Required runtime dependencies moved to `unisim-core>=1.7.15`,
  `unilab-rl==1.4.11`, and `mjbatch-uni~=0.2.5`; SAC uses the upstream SAC
  learner naming (`algo_log_name=sac`)
  ([#2071](https://github.com/Motphys/UniLab/pull/2071),
  [#2074](https://github.com/Motphys/UniLab/pull/2074),
  [#2112](https://github.com/Motphys/UniLab/pull/2112),
  [#2115](https://github.com/Motphys/UniLab/pull/2115)).
  必需运行时依赖升级为 `unisim-core>=1.7.15`、`unilab-rl==1.4.11`、
  `mjbatch-uni~=0.2.5`；SAC 使用上游 SAC learner 命名
  （`algo_log_name=sac`）。

### Added / 新增

- Added a Manager-owned selected-reset lifecycle: tensor reset rows, an owner
  reset transaction, reset-row observation publication, device-resident reset
  randomization negotiated through `device_reset_randomization`, and
  startup-mode body mass/COM randomization with committed baseline promotion.
  新增 Manager-owned selected-reset 生命周期：tensor reset row、owner reset
  transaction、reset-row observation publication、通过
  `device_reset_randomization` 协商的 device-resident reset DR，以及带已提交
  baseline promotion 的 startup-mode body mass/COM DR。

- Added scene-owned packed tensor reads for entity joint, body, and named
  sensor state; Manager terms consume validated device views, stable aggregate
  views, and backend-declared DEVICE_RESIDENT sensor inventories instead of
  hardcoded backend tables. Missing required device sensors fail closed.
  新增 scene-owned packed tensor read：覆盖 entity joint/body/named sensor
  状态；Manager term 消费经过校验的 device view、稳定 aggregate view 与后端
  声明的 DEVICE_RESIDENT sensor inventory，而不是硬编码后端表。缺失的必需
  device sensor 会 fail closed。

- Added Manager-owned Torch RNG and deterministic seed/reset behavior for
  observation noise, commands, events, impulses, and selected reset. Device
  noise, temporal/delay state, and scalar sensor observation carriers are
  tensor-native.
  新增 Manager-owned Torch RNG，覆盖观测噪声、command、event、impulse 与
  selected reset 的 seed/reset 行为。噪声、temporal/delay 状态和标量 sensor
  observation carrier 均为 tensor-native。

- Added the CUDA MPS single-rank execution-sharing mode and the user-owned
  `uni-cumps` CLI (`status`, `doctor`, `start`, `stop`, `env`). Training
  requests sharing explicitly with `training.cuda_process_sharing=mps` and
  validates a recorded daemon before construction; it never silently starts or
  stops a host service
  ([ADR-0013](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0013-cuda-mps-single-rank-execution-sharing.md),
  [ADR-0014](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0014-cuda-mps-cli-lifecycle-owner.md),
  [#2065](https://github.com/Motphys/UniLab/pull/2065),
  [#2069](https://github.com/Motphys/UniLab/pull/2069),
  [#2085](https://github.com/Motphys/UniLab/pull/2085)).
  新增 CUDA MPS single-rank execution-sharing 模式与用户拥有的 `uni-cumps`
  CLI（`status` / `doctor` / `start` / `stop` / `env`）。训练必须通过
  `training.cuda_process_sharing=mps` 显式请求，并在构造前校验已记录 daemon；
  训练进程不会隐式启动或停止 host 服务（ADR-0013、ADR-0014）。

- Added tensor-runtime production soak tooling, replay-ingress diagnostics,
  abnormal-shutdown evidence, schema/bounds consumption, tensor NaN guards, and
  the production guide for the scoped G1 Motion Tracking / FlashSAC / MJWarp
  path.
  新增 tensor-runtime 生产 soak 工具、replay-ingress 诊断、异常退出证据、
  schema/bounds 消费、tensor NaN guard，以及 scoped G1 Motion Tracking /
  FlashSAC / MJWarp 路径的生产指南。

- Added opt-in off-policy training resume (`algo.resume=true`), explicit
  playback checkpoint selection, FlashSAC `policy_before_critic`, learner
  whole-cycle CUDA Graph controls, and grouped terminal metrics with bounded
  width tiers.
  新增 opt-in off-policy 训练续训（`algo.resume=true`）、显式 playback
  checkpoint 选择、FlashSAC `policy_before_critic`、learner whole-cycle CUDA
  Graph 控制，以及带宽度分层的分组终端指标。

- Added generic motion-tracking capabilities retained for downstream task
  packages: universal combo rewards, optional motion NPZ torque fields, tensor
  motion sampler/carriers, initialized physical action targets with public
  `tensor_target`, Gaussian-noise clamp, recursive hashable keys for shared
  cross-group observation terms, and MJWarp `set_state_tensor` benchmark
  timings/profile syntax.
  新增供下游任务包复用的 generic motion-tracking 能力：universal combo
  reward、可选 motion NPZ torque 字段、tensor motion sampler/carrier、确定性
  初始化且公开 `tensor_target` 的物理 action target、Gaussian noise clamp、
  跨组共享 observation term 的递归 hashable key，以及 MJWarp
  `set_state_tensor` benchmark timing/profile 语法。

### Changed / 变更

- Reduced Manager host boundaries through packed reads, stable device views,
  fused action/reward/termination/observation/reset kernels, device-side
  samplers and RNG, cached tensor bounds, delayed reward host publication, and
  phase-level update-state/reset timing attribution.
  通过 packed read、稳定 device view、融合 action/reward/termination/
  observation/reset kernel、device-side sampler 与 RNG、缓存 tensor bounds、
  延迟 reward host publication，以及 update-state/reset 的阶段级 timing
  attribution，减少 Manager host boundary。

- MuJoCo remains the canonical HOST_BRIDGE backend. Its default Manager carrier
  stays on CPU even when a learner resolves to CUDA; an indexed CUDA collector
  carrier remains an explicit `training.collector_tensor_device=cuda` request.
  MuJoCo 仍是 canonical HOST_BRIDGE backend。即使 learner 解析到 CUDA，其默认
  Manager carrier 也保持 CPU；indexed CUDA collector carrier 必须显式设置
  `training.collector_tensor_device=cuda`。

- Removed the repository-wide uv version pin and kept lockfile reproducibility;
  CI and Docker use the standard uv installation path.
  移除仓库级 uv 版本限制；依赖可复现性仍由 lockfile 提供，CI 与 Docker 使用
  标准 uv 安装路径。

### Fixed / 修复

- Moved MimicLite/FlashMimic task ownership downstream: UniLab no longer ships
  the three `mjwarp_mimiclite*` owners, task torque-sensor scene, future-window
  observation module, or MimicLite applied-torque term. Generic tensor runtime
  capabilities remain upstream
  ([#2119](https://github.com/Motphys/UniLab/issues/2119),
  [#2120](https://github.com/Motphys/UniLab/pull/2120)).
  MimicLite/FlashMimic 任务所有权迁至下游：UniLab 不再内置三个
  `mjwarp_mimiclite*` owner、任务 torque-sensor 场景、future-window observation
  module 或 MimicLite applied-torque term；generic tensor runtime 能力保留在上游。

- Fixed frozen full-refresh motion references, partial tensor-command probing,
  stale device-resident joint reward state, selected-reset readiness boundaries,
  initial episode counter staggering, reset-row observation scale slicing, and
  device sensor fallback behavior.
  修复 full refresh 中 motion reference 冻结、partial tensor-command probing、
  joint reward 使用 stale device-resident state、selected-reset readiness
  boundary、初始 episode counter 交错、reset-row observation scale 切片，以及
  device sensor fallback 行为。

- Fixed host-bridge Manager Torch device handling, explicit device validation,
  APPO/MJWarp test stability, ONNX export device examples, IsaacSim mapped
  owner cold/dtype/readiness contracts, and Motrix owner restoration.
  修复 host-bridge Manager Torch device 处理、显式设备校验、APPO/MJWarp 测试
  稳定性、ONNX export device 示例、IsaacSim mapped owner 的 cold/dtype/
  readiness 契约，以及 Motrix owner 恢复。

## 1.3.3 (2026-09-27)

### Changed / 变更

- Removed the repository-wide uv version pin: projects no longer require
  `uv==0.12.5`, CI and Docker use the standard uv installation path, and
  dependency reproducibility remains provided by `uv.lock`
  ([#1809](https://github.com/Motphys/UniLab/issues/1809)).
  移除仓库级 uv 版本限制：项目不再要求 `uv==0.12.5`，CI 与 Docker 使用
  标准 uv 安装路径；依赖可复现性仍由 `uv.lock` 提供（#1809）。

### Added / 新增

- Added the WarpSAC off-policy algorithm shipped by unilab-rl 1.4.0: the
  `conf/warpsac` owner config tree with G1 owners for `g1_walk_flat` and
  `g1_motion_tracking` on the MuJoCo and mjwarp backends, the
  `train_warpsac.py` entrypoint, CLI and documentation wiring, and a
  stabilized G1 walk replay profile; explicit WarpSAC playback run selection
  is now honored
  ([#1645](https://github.com/Motphys/UniLab/pull/1645)).
  新增 unilab-rl 1.4.0 提供的 WarpSAC off-policy 算法：`conf/warpsac` owner
  配置树，覆盖 MuJoCo 与 mjwarp 后端的 `g1_walk_flat` 与
  `g1_motion_tracking` G1 owner、`train_warpsac.py` 入口、CLI 与文档接线，
  以及稳定化的 G1 walk 回放 profile；显式指定的 WarpSAC 回放 run 现在会被
  正确遵循（#1645）。

- The viser browser-based viewer is now a required dependency and works on
  every playback-capable backend: `eval --render-mode viser` and
  `training.play_render_mode=viser` route through the shared playback loop
  for mujoco/mjwarp/newton/drake/superdex
  ([#1631](https://github.com/Motphys/UniLab/issues/1631),
  [#1632](https://github.com/Motphys/UniLab/pull/1632),
  [#1638](https://github.com/Motphys/UniLab/pull/1638)), and the Motrix
  backend gained viser rendering through the unisim-core 1.7.6 physics-state
  playback contract (portable entity scenes and audited whole-MJCF scenes)
  plus the 1.7.8 `<worldbody>` merge fix
  ([#1639](https://github.com/Motphys/UniLab/pull/1639)). The `viser.*`
  config group (port, max_envs, display_mode, env_idx) is now available on
  the appo/sac/flashsac configs to match ppo.
  viser 浏览器查看器成为必需依赖，并支持所有具备回放能力的后端：
  `eval --render-mode viser` 与 `training.play_render_mode=viser` 通过共享
  回放循环支持 mujoco/mjwarp/newton/drake/superdex（#1631、#1632、#1638）；
  Motrix 后端借助 unisim-core 1.7.6 的 physics-state 回放契约（portable
  实体场景与经审计的 whole-MJCF 场景）及 1.7.8 的 `<worldbody>` 合并修复
  获得 viser 渲染能力（#1639）。`viser.*` 配置组（port、max_envs、
  display_mode、env_idx）在 appo/sac/flashsac 配置上对齐 ppo 提供。

- Added the IsaacSim PhysX GPU contact buffer knobs
  `isaacsim_gpu_max_rigid_contact_count` and
  `isaacsim_gpu_max_rigid_patch_count` to `EnvCfg`, forwarded with fail-fast
  validation to the UniSim factory and omitted when unset so older
  unisim-core releases fail closed instead of seeing unknown kwargs
  ([#1636](https://github.com/Motphys/UniLab/pull/1636)).
  `EnvCfg` 新增 IsaacSim PhysX GPU 接触缓冲旋钮
  `isaacsim_gpu_max_rigid_contact_count` 与
  `isaacsim_gpu_max_rigid_patch_count`，以 fail-fast 校验转发给 UniSim
  factory；未设置时省略该键，旧版 unisim-core 不会收到未知 kwargs，而是
  fail closed（#1636）。

- Added the `training.log_interval` key (default 1, no behavior change) and
  batched TensorBoard scalar logging for rsl_rl PPO: each iteration's
  `add_scalar` calls become a single event record gated by the interval,
  removing the writer-queue stall on network filesystems
  ([#1646](https://github.com/Motphys/UniLab/issues/1646),
  [#1648](https://github.com/Motphys/UniLab/pull/1648)).
  新增 `training.log_interval` 配置键（默认 1，不改变行为）与 rsl_rl PPO 的
  TensorBoard 标量批量写入：每次迭代的 `add_scalar` 调用合并为单条 event
  record 并按间隔落盘，消除网络文件系统上的写入队列阻塞（#1646、#1648）。

- Documented and tested the Motrix `cpu_ids` worker-affinity wiring, and
  added a durable A/B microbenchmark for the fused body-state read (fused
  packed read 5.7x, caller-owned-out copy 11.7x over the legacy cache-based
  copy at 8192 envs x 14 tracked bodies; end-to-end Motrix runs show no
  regression, so the single fused path is kept)
  ([#962](https://github.com/Motphys/UniLab/issues/962),
  [#1308](https://github.com/Motphys/UniLab/issues/1308),
  [#1634](https://github.com/Motphys/UniLab/pull/1634),
  [#1635](https://github.com/Motphys/UniLab/pull/1635)).
  记录并测试 Motrix `cpu_ids` worker 亲和性接线，并为 fused body-state
  读取新增持久化 A/B 微基准（8192 envs x 14 tracked bodies 下 fused
  packed read 快 5.7 倍、caller-owned-out copy 快 11.7 倍；端到端 Motrix
  运行无回归，保留单一 fused 路径）（#962、#1308、#1634、#1635）。

### Changed / 变更

- Raised the base UniSim requirement to `unisim-core>=1.7.8` and mjbatch-uni
  to `~=0.2.4`. Interactive rendering now consumes the unisim 1.7.5
  physics-state playback contract through
  `unilab.visualization.playback_state`: snapshots split through the
  physics-state layout contract instead of hardcoded
  `mjSTATE_FULLPHYSICS` sizes, mocap geometry replays through
  `get_playback_mocap_state`, and playback entrypoints precheck
  `supports_physics_state_playback` and fail closed with a targeted message
  ([#1637](https://github.com/Motphys/UniLab/pull/1637),
  [#1639](https://github.com/Motphys/UniLab/pull/1639)).
  基础 UniSim 依赖提升到 `unisim-core>=1.7.8`，mjbatch-uni 提升到
  `~=0.2.4`。交互式渲染通过 `unilab.visualization.playback_state` 消费
  unisim 1.7.5 的 physics-state 回放契约：快照按 physics-state 布局契约
  拆分，不再硬编码 `mjSTATE_FULLPHYSICS` 尺寸；mocap 几何通过
  `get_playback_mocap_state` 回放；回放入口预检
  `supports_physics_state_playback` 并以明确信息 fail closed
  （#1637、#1639）。

- Bumped the optional `uni_rl` extra and dev environment from unilab-rl
  1.3.2 to 1.4.3: 1.4.0 delivers WarpSAC, off-policy metric consumers
  migrated to the 1.4.1 canonical metric schema, the off-policy
  inference request timeout config was removed, and 1.4.3 stops the
  spurious "Learner stopped before inference tick" collector crash after
  normal off-policy training completion
  ([#1640](https://github.com/Motphys/UniLab/pull/1640),
  [#1643](https://github.com/Motphys/UniLab/pull/1643),
  [#1645](https://github.com/Motphys/UniLab/pull/1645),
  [#1655](https://github.com/Motphys/UniLab/pull/1655),
  [#1658](https://github.com/Motphys/UniLab/pull/1658),
  [#1659](https://github.com/Motphys/UniLab/pull/1659),
  [#1662](https://github.com/Motphys/UniLab/pull/1662)).
  可选 `uni_rl` extra 与 dev 环境的 unilab-rl 从 1.3.2 升级到 1.4.3：1.4.0
  提供 WarpSAC；off-policy 指标消费方迁移到 1.4.1 的规范指标 schema；移除
  off-policy 推理请求超时配置；1.4.3 修复了 off-policy 训练正常结束后
  collector 误报 "Learner stopped before inference tick" 崩溃的问题
  （#1640、#1643、#1645、#1655、#1658、#1659、#1662）。

- Bumped torch to 2.14 and unified the CUDA wheels on cu130 on Linux and
  Windows; measured on Apple Silicon, FlashSAC end-to-end throughput
  improves by 35%
  ([#1647](https://github.com/Motphys/UniLab/pull/1647)). ONNX policy
  export now defaults to opset 18, matching the onnxscript function
  library torch 2.14's dynamo exporter emits
  ([#1654](https://github.com/Motphys/UniLab/issues/1654),
  [#1656](https://github.com/Motphys/UniLab/pull/1656)).
  torch 升级到 2.14，Linux 与 Windows 的 CUDA wheel 统一到 cu130；Apple
  Silicon 上实测 FlashSAC 端到端吞吐提升 35%（#1647）。ONNX 策略导出默认
  opset 调整为 18，与 torch 2.14 dynamo 导出器生成的 onnxscript 函数库
  匹配（#1654、#1656）。

- Enabled full-objective compile for FlashSAC
  ([#1641](https://github.com/Motphys/UniLab/pull/1641)), and pinned uv to
  0.12.5 across the project, CI, and Docker
  ([#1657](https://github.com/Motphys/UniLab/pull/1657)).
  FlashSAC 启用 full-objective compile（#1641）；uv 在工程、CI 与 Docker
  中统一固定到 0.12.5（#1657）。

### Fixed / 修复

- The motion CSV loader preserves headerless frames
  ([#1653](https://github.com/Motphys/UniLab/pull/1653)).
  动作 CSV 加载器保留无表头的帧（#1653）。

- macOS MuJoCo interactive eval now runs through mjpython (Cocoa main
  thread), and playback exits directly afterward to avoid the
  free-threaded CPython 3.13 interpreter-finalization deadlock
  ([#1629](https://github.com/Motphys/UniLab/issues/1629),
  [#1630](https://github.com/Motphys/UniLab/pull/1630)).
  macOS 上 MuJoCo 交互式 eval 改经 mjpython 运行（Cocoa 主线程），回放
  结束后直接退出，避免 free-threaded CPython 3.13 解释器收尾死锁
  （#1629、#1630）。

- The viser viewer batches scene updates and shuts down cleanly on Ctrl-C
  ([#1633](https://github.com/Motphys/UniLab/pull/1633)).
  viser 查看器批量提交场景更新，并在 Ctrl-C 时干净退出（#1633）。

- Play artifact failures are decoupled: ONNX export and video rendering run
  under a shared nonfatal guard across the off-policy, APPO, and rsl-rl
  play paths, so one artifact failure no longer blocks the other
  ([#1654](https://github.com/Motphys/UniLab/issues/1654),
  [#1656](https://github.com/Motphys/UniLab/pull/1656)).
  回放产物失败解耦：ONNX 导出与视频渲染在 off-policy、APPO 与 rsl-rl
  回放路径上统一走非致命保护，单个产物失败不再阻塞另一个
  （#1654、#1656）。

- Restored the ROCm runtime dependencies and covered SAC cold-start
  inference compilation
  ([#1650](https://github.com/Motphys/UniLab/pull/1650)); the CI test job
  no longer rolls git-tracked robot scene XMLs back to the cached snapshot
  revision after restoring the asset cache
  ([#1639](https://github.com/Motphys/UniLab/pull/1639)).
  恢复 ROCm 运行时依赖，并覆盖 SAC 冷启动推理编译（#1650）；CI 测试任务
  恢复资产缓存后不再将 git 跟踪的机器人场景 XML 回滚到缓存快照版本
  （#1639）。

## 1.3.2 (2026-09-22)

### Breaking changes / 破坏性变更

- Removed the TD3 algorithm: the `conf/td3` owner config tree, the
  `train_td3.py` entrypoint, the TD3 structured configs, and the TD3 dispatch
  branches in training, playback, and the CLI
  ([#1625](https://github.com/Motphys/UniLab/pull/1625)). The optional
  `unilab-rl` dependency moves to 1.3.2, the release that removed
  `uni_rl.algos.fast_td3`. Use SAC or FlashSAC for off-policy training.
  移除 TD3 算法：`conf/td3` owner 配置树、`train_td3.py` 入口、TD3 结构化
  配置以及训练、回放与 CLI 中的 TD3 分发分支（#1625）。可选 `unilab-rl`
  依赖升级到移除了 `uni_rl.algos.fast_td3` 的 1.3.2 版本。off-policy 训练
  请改用 SAC 或 FlashSAC。

### Added / 新增

- Added SAC owners for `g1_motion_tracking` on the Genesis, IsaacGym, Newton,
  and IsaacSim backends, with per-backend domain-randomization splits narrowed
  to each backend's declared capability set
  ([#1618](https://github.com/Motphys/UniLab/pull/1618),
  [#1619](https://github.com/Motphys/UniLab/pull/1619)).
  为 `g1_motion_tracking` 新增 Genesis、IsaacGym、Newton、IsaacSim 后端的
  SAC owner，并按后端拆分域随机化，收窄到各后端声明的 capability 集合
  （#1618、#1619）。

- Added IsaacSim PhysX solver knobs to `EnvCfg`
  (`isaacsim_solver_position_iteration_count`,
  `isaacsim_solver_velocity_iteration_count`,
  `isaacsim_bounce_threshold_velocity`, `isaacsim_contact_offset`,
  `isaacsim_rest_offset`, `isaacsim_max_depenetration_velocity`), forwarded
  by `env_backend_kwargs` to the bounded, readback-validated solver
  configuration UniSim 1.7.x exposes on the IsaacSim backend; `None` keeps
  the PhysX scene defaults
  ([#1617](https://github.com/Motphys/UniLab/issues/1617)).
  `EnvCfg` 新增 IsaacSim PhysX 求解器旋钮
  （`isaacsim_solver_position_iteration_count`、
  `isaacsim_solver_velocity_iteration_count`、
  `isaacsim_bounce_threshold_velocity`、`isaacsim_contact_offset`、
  `isaacsim_rest_offset`、`isaacsim_max_depenetration_velocity`），由
  `env_backend_kwargs` 转发至 UniSim 1.7.x 在 IsaacSim 后端暴露的有界、
  带读回验证的求解器配置；`None` 保持 PhysX 场景默认值
  ([#1617](https://github.com/Motphys/UniLab/issues/1617)）。

### Changed / 变更

- PPO now drives upstream `rsl-rl-lib` directly through the UniLab-owned
  VecEnv adapter and distributed helpers; `uni_rl` (the `unilab-rl` package)
  is an optional extra needed only for APPO, off-policy algorithms, and
  multi-GPU data-parallel PPO launches
  ([#1621](https://github.com/Motphys/UniLab/issues/1621),
  [#1622](https://github.com/Motphys/UniLab/pull/1622)). Single-process PPO
  training and playback work without `uni_rl` installed.
  PPO 改为通过 UniLab 自己的 VecEnv adapter 与分布式 helper 直接驱动上游
  `rsl-rl-lib`；`uni_rl`（`unilab-rl` 包）变为可选 extra，仅 APPO、
  off-policy 算法与多卡数据并行 PPO 启动需要（#1621、#1622）。单进程 PPO
  训练与回放不再依赖 `uni_rl`。

- Raised the base UniSim requirement to `unisim-core>=1.7.4` and pinned
  `unilab-rl==1.3.2` for the optional `uni_rl` extra and the dev environment;
  the MuJoCo backend uses the exact tracked-body-state path from the executor.
  基础 UniSim 依赖提升到 `unisim-core>=1.7.4`，可选 `uni_rl` extra 与 dev
  环境固定 `unilab-rl==1.3.2`；MuJoCo 后端改用执行器的精确
  tracked-body-state 路径。

## 1.3.1 (2026-09-20)

### Added / 新增

- Add the M2 entity consumer: typed physical sources/variants, explicit logical-to-physical binding, selected entity reset transactions and per-variant state/control defaults. A registered primitive task validates the same pickleable EnvFactory with MuJoCo and IsaacSim. Standard and ROCm dependency profiles require the released `unisim-core>=1.5.0` package, without a Git source override. Regression tests use the unified worker and construct the nonfirst-root fixture through the public factory.
  新增 M2 实体消费层：物理源/variant 类型化、显式逻辑到物理绑定、局部实体 reset 事务和逐 variant 状态/控制默认值。注册 primitive task 验证同一可 pickle EnvFactory 在 MuJoCo/IsaacSim 的行为。标准与 ROCm 依赖配置要求已发布的 `unisim-core>=1.5.0` 包，不使用 Git source 覆盖。回归测试使用统一 worker，并通过公共 factory 构造非首 free root 场景。
- Remove quadratic selected-row lookup and unused full-batch snapshots from entity reset staging. Reject mapped logical roots that point at descendant bodies, keeping reads, defaults and writes aligned. A bounded A/B script records separate row-index and sparse-field effects without claiming simulation throughput.
  删除实体 reset 暂存中的平方级行查找和无用全批状态快照。拒绝指向后代 body 的逻辑 root 绑定，使读取、默认值和写入保持一致。限定规模的 A/B 脚本分别记录行索引与稀疏字段的影响，不宣称仿真吞吐提升。

- Added IsaacGym fixed-variant protocol and real-runtime coverage. The
  deterministic worker mock validates and echoes the construction-time variant
  assignment, while the external Preview-4 slow lane realizes per-env actor
  asset selection and per-variant keyframes; public layout drift fails closed.
  IsaacGym fixed variants and the still-pending 600-variant support decision
  are documented on the backend page.
  新增 IsaacGym fixed-variant 协议层与真实 runtime 覆盖。确定性 worker mock
  校验并回显 construction-time variant assignment；外部 Preview 4 slow lane
  验证逐环境 actor 资产选择与逐变体 keyframe；公共布局漂移 fail closed。
  IsaacGym fixed variants 及仍待决策的 600 变体支持边界已写入后端文档。
- Added the IsaacGym fixed-variant scale benchmark. Each variant count runs in
  a fresh child process and records source generation, construction time,
  live worker RSS, and control/physics/env-step rates; results are written as
  a versioned JSON artifact for the #1579 support decision.
  新增 IsaacGym fixed-variant 规模 benchmark。每个 variant 数在独立子进程中
  运行，记录源生成、构造时间、worker 实时 RSS 与 control/physics/env-step
  速率，并输出版本化 JSON artifact 供 #1579 support 决策使用。

### Changed / 变更

- Moved the Motrix runtime pin into UniSim. The `motrix` extra now resolves
  `unisim-core[motrix]>=1.7.3` instead of pinning `motrixsim-core==0.8.2`
  directly, so the consumed Motrix runtime always matches the version the
  `unisim.backend.motrix` adapter is tested against (currently
  motrixsim-core 0.10.1).
  将 Motrix runtime 的版本固定移入 UniSim。`motrix` extra 现在解析
  `unisim-core[motrix]>=1.7.3`，不再直接固定 `motrixsim-core==0.8.2`，
  使消费的 Motrix runtime 始终与 `unisim.backend.motrix` 适配层的测试
  版本一致（当前为 motrixsim-core 0.10.1）。

- Raised the base and optional SuperDex UniSim requirements to
  `unisim-core>=1.7.3`. The released package adds IsaacSim mapped-scene
  per-body net contact force/found sensors, bounded PhysX solver
  configuration, per-entity self-collision and mapped reset domain
  randomization, plus IsaacGym mapped-scene reset randomization and interval
  body wrenches (UniSim #248/#249/#251/#255).
  将基础与可选 SuperDex 的 UniSim 依赖提升到 `unisim-core>=1.7.3`。已发布
  包新增 IsaacSim mapped scene 的逐 body 净接触力/found 传感器、有界
  PhysX solver 配置、逐实体 self-collision 与 mapped reset 域随机化，以及
  IsaacGym mapped scene 的 reset 域随机化和 interval body 力矩
  （UniSim #248/#249/#251/#255）。

- Raised the base and optional SuperDex UniSim requirements to
  `unisim-core>=1.7.2`. The released package adds the SuperDex portable-scene
  physical-root integration, including physical kinematic roots, portable
  collision-disabled mirrors and scoped selected-control reset restoration
  (UniSim #154).
  将基础与可选 SuperDex 的 UniSim 依赖提升到 `unisim-core>=1.7.2`。已发布
  包新增 SuperDex portable scene 的物理 root 集成，包括物理 kinematic root、
  portable collision-disabled mirror 和限定范围的 selected-control reset
  恢复（UniSim #154）。
- Raised the base and optional SuperDex UniSim requirements to
  `unisim-core>=1.7.1`. The released packages contain portable MJCF profile
  v1 and the IsaacSim worker dependency-isolation fix, so consumer acceptance
  no longer uses the provisional local-checkout skip and validates the
  robot/object/table/mirror fixture from PyPI dependencies (#1609; UniSim
  #154/#155).
  将基础与可选 SuperDex 的 UniSim 依赖提升到 `unisim-core>=1.7.1`。已发布
  包包含 portable MJCF profile v1 与 IsaacSim worker 依赖隔离修复，因此
  消费验收不再使用临时本地 checkout skip，并基于 PyPI 依赖验证
  robot/object/table/mirror fixture（#1609；UniSim #154/#155）。
- Raised the base UniSim requirement to `unisim-core>=1.4.1` to consume the
  published IsaacGym fixed-variant adapter. The optional `superdex` extra's
  own `>=1.4.0` constraint is unchanged: the base requirement already forces
  every installed profile to 1.4.1 or newer, and SuperDex has no 1.4.1-specific
  dependency change. The UniLab package version is unchanged.
  将基础 UniSim 依赖提升到 `unisim-core>=1.4.1`，以消费已发布的 IsaacGym
  fixed-variant adapter。可选 `superdex` extra 自身的 `>=1.4.0` 约束保持不变：
  基础依赖已经强制所有安装 profile 使用 1.4.1 或更新版本，且 SuperDex 在
  1.4.1 中没有专属依赖变化。UniLab 包版本保持不变。

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
