---
orphan: true
---

# ADR-0016 RL Reintegration Into unilab.rl And Single Distribution

语言: 简体中文

- Status: Proposed
- Date: 2026-10-16
- Update: 2026-10-19 —— 下文“保留空 extra”的过渡安排已执行完毕，
  `uni_rl` extra 及 `--extra uni_rl` 安装入口已移除。
- Owners: RL / Runtime / Packaging maintainers
- Supersedes: None（替代 roadmap #1476 确立的 unilab-rl 独立发行决策；该决策当时未记录为 ADR）
- Superseded by: None

## Context

UniLab 1.3.0 前后将 RL 算法与异步 runtime 拆分为独立仓库 `unilabsim/unilab_rl`
（distribution `unilab-rl`，import namespace `uni_rl`），UniLab 经可选 extra `uni_rl`
依赖它。拆分后同一契约分属两侧，靠复制或跨包委托维持一致，产生 #1623 记录的双实现
漂移：

- `UNILAB_DP_LOG_DIR`、torchrun rank/local-rank/world-size reader、removed-device 配置
  检查在 `unilab.rl.distributed` 与 `uni_rl.ipc.dp_launcher` 各有一份定义；
- `resolve_collector_cpu_ids`、`launch_torchrun_workers` 在 UniLab 侧只是经
  `_require_uni_rl_dp_launcher` 的懒加载转发，签名靠两个独立版本维持对齐；
- `CUDA_VISIBLE_DEVICES` 解析在 `unilab.base.process_device` 与 `uni_rl.ipc.dp_launcher`
  各有一份；
- 训练 seed 逻辑在 `unilab.training.seed` 与 `uni_rl.utils.seed` 平行重复，且对 NumPy
  RNG 的副作用不一致。

仅合并 Git 仓库、保留两个包和两套实现不能消除该问题；解决标准是唯一实现及其行为
测试。审计基线：UniLab `5b27e5d9`（1.3.5）、unilab_rl `3654e4b`（1.4.15），方案细节见
roadmap issue #2372。

## Decision

采用**一个仓库、一个 `unilab` distribution、一个 `unilab.rl` 算法与训练执行子系统**。
`unisim-core` 继续作为独立物理层（ADR-0007 不变）。PPO 的 runner、algorithm 与
collective 继续由上游 rsl-rl 提供，不建立自研 `algos/ppo`。

### 目标分层与 owner

`src/unilab/rl/` 按语义分五层：

- `rl/contracts/`：env 注入契约（`EnvProtocol`/`EnvFactory`/capabilities）、obs/critic 与
  final-observation 语义、冷路径环境契约探测。只依赖类型与通用 tensor 库，不导入具体
  环境、registry、manager、backend 或算法。
- `rl/adapters/rsl_rl/`：第三方协议适配。`vec_env.py` 持有唯一 `RslRlVecEnvAdapter`；
  `distributed.py` 持有 PPO 专属 rank seed/device/样本数/结束/回放拓扑助手。
- `rl/algos/{appo,sac,flash_sac,common}/`：自研算法 runner/learner/网络与算法专属
  staging/replay 策略。可依赖 contracts、runtime、logging 与 rsl-rl，不构造具体 UniLab
  环境。
- `rl/runtime/`：训练执行。`distributed/{context,launcher,affinity,sync}`、`ipc/`（仅进程间
  数据与同步机制）、`offpolicy/`、`seed.py`、`device.py`、`settings.py`、`placement.py`、
  `tensor.py`、`nan_guard.py`。可依赖 contracts 与 RL logging；不依赖具体算法、
  `training`、任务注册或 simulator，算法经 learner/builder/callback 注入。
- `rl/logging/`：RL logger、metric schema、trace。

相邻层职责：`training/` 管实验装配与落盘（含 Hydra rank 配置转换、removed-device 配置
拒绝、plain JSON 训练状态 checkpoint 序列化/复制），`base/process_device.py` 管
simulator 专属设备绑定。`CUDA_VISIBLE_DEVICES` 的
通用 token 解析下沉到 stdlib-only 的 `unilab/process_context.py`，供 base 与 RL 共用；
backend payload/device namespace 翻译留在 base。runtime 不反向 import training。

### #1623 符号唯一属主

| 契约/行为 | 唯一属主 |
|---|---|
| `UNILAB_DP_LOG_DIR/RANK/WORLD_SIZE`、torchrun rank reader | `rl.runtime.distributed.context` |
| `CUDA_VISIBLE_DEVICES` 通用 token 解析 | `unilab.process_context`（stdlib-only） |
| `launch_torchrun_workers`、进程监督 | `rl.runtime.distributed.launcher` |
| `resolve_collector_cpu_ids` | `rl.runtime.distributed.affinity` |
| removed-device 配置拒绝、Hydra rank 配置转换 | training 配置装配层 |
| PPO rank seed、rsl device、每轮样本数、结束与回放拓扑 | `rl.adapters.rsl_rl.distributed` |
| `RslRlVecEnvAdapter` 与第三方 obs-dim 映射 | `rl.adapters.rsl_rl.vec_env` |
| obs/可选 critic、terminal observation 规则 | `rl.contracts.observations` |

旧转发/复制路径删除；若迁移过渡期保留原 `unilab.rl` 导出面，只允许同一对象的
re-export，禁止带独立签名/默认值/转换逻辑的第二层实现。

### 包初始化与导入闭包

`unilab`、`unilab.rl`、`rl.runtime`、`rl.runtime.distributed` 的 `__init__.py` 必须轻量：
不聚合导入算法、Torch、TensorDict，导入 launcher/context 不得触发 CUDA 初始化或任务
注册。边界守卫由 `tests/test_library_import_boundary.py` 改为内部 DAG 检查：RL 核心不
依赖 base/envs/tasks/managers/training/scripts/unisim，runtime 不反向依赖 algos。

### 发行与兼容

- 最终只发布 `unilab`：标准与 ROCm pyproject、dev group、Makefile、CI 与锁文件移除
  `unilab-rl` 依赖；依赖约束按合并需要合并（纳入 `wandb>=0.22.3`），不取两边并集。
- `unilab[uni_rl]` 保留为空 extra 作为旧安装命令的过渡兼容，文档标注其无必要并设定
  移除版本；不承诺永久支持 `import uni_rl`。若未来证实必须支持旧 import，另行设计有限期
  compatibility distribution，禁止与旧实现同时提供两份类定义。
- 外部扩展点保持有效：`runtime_resolver`、`class_name`、`actor_adapter_modules`、actor
  registry 与 warmup hook 继续存在，内置路径更新到新 namespace，外部包路径保持原值；
  `OffPolicyRuntime` 的 isinstance 校验要求任何兼容出口重导出同一个 class。
- 保持日志 metric schema 与 checkpoint schema；新 provenance 记录统一 UniLab 版本与迁移
  来源，历史 `unilab_rl` 版本字段允许读取。历史 checkpoint 仅在旧路径解析边界做显式
  映射，不全局重写历史 run config。
- **本次开发不改动项目版本号**：标准/ROCm pyproject、运行时版本声明与锁文件中的本项目
  版本保持原值；不打 release tag、不发布 PyPI。发版是后续独立工作。

### 分支治理

- 最终 PR base：`main`；集成分支 `dev/issue-2372-rl-reintegration`。
- child 分支基于集成分支、PR 只合入集成分支；统一 head 验收后再向 main 提 PR。
- 源码统一与撤包作为同一可交付单元验收，避免 main 处于需同时安装两套实现的状态。

## Stable Contracts

1. 上表每个 #1623 契约只有一个定义；跨进程行为测试覆盖 parent 写 log-dir/rank → spawn
   child 读取、失败传播与清理，而非只比较常量字符串。
2. rank/visibility 测试覆盖单卡、多卡、UUID/MIG token 与 rank-local `cuda:0`，不把
   LOCAL_RANK 当作 CUDA 索引；CPU 分区测试覆盖单 rank、非连续 affinity、SMT、显式
   重叠/空分区与非法 rank。
3. observation adapter 用同一组输入验证 PPO 与 APPO 回放（actor/critic 维度、顺序、
   dtype/device、terminated/truncated、time_outs、episode state、训练状态导入导出）；
   观测语义遵循 ADR-0005、ADR-0011/0012，迁移不得静默改变模型输入。
4. 不安装 `unilab-rl` 时，统一测试与隔离 wheel/sdist 安装仍可完成相关导入与最小训练
   路径；`py.typed` 与 `native_h2d_ext.cpp` 进入 wheel/sdist。
5. 两 GPU PPO smoke（启动、rank 0 落盘、失败传播/清理、训练后回放）是 #1623 关闭条件；
   缺硬件证据时 issue 保留待验收状态。
6. RL CPU 单测在独立子进程验证，避免根 conftest 的 registry 注入掩盖环境独立性；合并
   coverage 保留原 RL 覆盖强度。

## Alternatives Considered

- 维持两个仓库、靠签名比对对齐接口：#1623 的根因（同一契约双实现）继续存在，拒绝。
- 仅合并 Git 仓库、保留 `uni_rl` 包与第二份实现：namespace 与发行版本仍双轨，漂移源
  未消除，拒绝。
- 在 `rl/algos/ppo` 建立自研 PPO：PPO runner/algorithm 属上游 rsl-rl 职责，自建会造成
  新的双实现，拒绝；`adapters/rsl_rl` 仅做协议适配。
- 迁移期保留 `unilab.rl.distributed` 转发层长期存在：转发层带独立签名与默认值即第二
  实现，拒绝；只允许纯 re-export 且限期移除。

## Consequences

- ADR-0011 中"direct rsl-rl/PPO 无 `uni_rl` 可运行、多卡启动才懒加载委托 `uni_rl.ipc`"
  与"`uni_rl` 不得 import UniLab"两条随本 ADR 失效：合并后 RL 与 env 同属一仓，边界
  改为 `unilab.rl` 内部 DAG（runtime 不依赖 algos、RL 核心不依赖 env/training 层）。
  ADR-0011 其余 contract（Torch-only manager runtime、tensor 生命周期）不变。
- `tests/test_library_import_boundary.py` 的"algos/ipc/logging 目录不得存在"与
  `_MIGRATED_SYMBOLS` 禁重定义清单被内部 DAG 守卫替代。
- 公开 API 路径变化（`uni_rl.*` → `unilab.rl.*`）必须记录迁移说明；CLI 算法名、日志
  路径、checkpoint 字段与 metric schema 不改名。
- 旧 `unilab_rl` 仓库停止独立维护与归档另行决定；其历史 commit/tag/release 保留，本仓
  记录来源 commit `3654e4b` 与版权，不重写 UniLab 历史。
- 以后不应再出现：跨包的同契约副本、经懒加载转发维持的"可选"核心 runtime、为关闭
  #1623 而恢复已删除的 `resolve_dp_topology`/`validate_dp_launchable` 接口。

## Evidence In Repo

- `src/unilab/rl/`：合并后的 contracts/adapters/algos/runtime/logging 分层。
- `src/unilab/process_context.py`：stdlib-only 进程可见性解析。
- `src/unilab/base/process_device.py`、`src/unilab/base/env_factory.py`：backend 设备绑定
  与可 pickle env factory。
- `tests/test_library_import_boundary.py`：内部 DAG 守卫。
- `tests/rl/`：迁移后的 RL 单测（contracts/adapters/algos/runtime/logging）。
- 默认/ROCm 依赖 profile 与对应 lock：单一 distribution 与空 `uni_rl` extra。
- Roadmap issue #2372、#1623、#1586；来源仓库 unilab_rl @ `3654e4b`（1.4.15）。

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`ADR-0001 Runtime Model And Layer Boundaries </adr/ADR-0001-runtime-model-and-layer-boundaries>`
- {doc}`ADR-0005 Unified Obs Critic Env And IPC Contract </adr/ADR-0005-unified-obs-critic-env-and-ipc-contract>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`ADR-0011 Torch-Only Manager-Based Runtime </adr/ADR-0011-torch-only-manager-based-runtime>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
