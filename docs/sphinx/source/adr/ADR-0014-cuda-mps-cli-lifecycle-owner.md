---
orphan: true
---

# ADR-0014 CUDA MPS CLI Lifecycle Owner

语言: 简体中文

- Status: Proposed
- Date: 2026-10-08
- Owners: UniLab training runtime maintainers
- Supersedes: [ADR-0013](ADR-0013-cuda-mps-single-rank-execution-sharing.md)（仅 daemon 生命周期管理边界）
- Superseded by: None

## Context

ADR-0013 把 CUDA MPS 定义为训练 owner 的显式 execution-sharing mode，并保持训练进程
只验证、不管理 daemon。该边界避免了训练进程误拥有共享 host 服务，但部署者仍需要手写
pipe/log 目录、daemon 启停、权限和 stale 进程处理脚本，且无法用结构化方式表达未来
multi-GPU 拓扑。

Issue #2079 要求提供一个显式、用户拥有的 CLI 生命周期面，同时不弱化训练路径的
fail-closed probe，也不提前声明 DP 或 task-per-GPU 支持。

## Decision

- 新增顶层 `uni-cumps` 命令，owner module 为
  `src/unilab/training/cuda_mps_cli.py`。
- 初始命令为 `status`、`doctor`、`start`、`stop`、`env`：
  - `status`/`doctor` 是只读诊断；
  - `start` 只启动显式请求的 user-owned daemon；
  - `stop` 只允许停止 UniLab record 证明拥有的 daemon；
  - `env` 输出 launcher 环境而不修改当前 shell。
- 单 GPU 单 daemon 是默认路径：未传 `--gpus` 时选择唯一可见物理 GPU；未传
  `--name` 时使用 `gpu-<full-uuid>`；未传 pipe/log 路径时使用 `/tmp/uni-cumps/<name>`（mode `0700`）；
  `env`/`stop` 未传 `--name` 时选择唯一 live record。多 GPU 或多 daemon 歧义必须
  显式指定，不自动选择。
- `stop` 接受 `--gpus <index-or-uuid>` 选择器（与 `--name`/`--all` 互斥），解析为
  覆盖该 GPU 的唯一 live record；零个或多个匹配 fail closed 并列出候选名称。
  stop authority 仍然只来自 record，不 attach 到 unmanaged daemon。
  （2026-11-16 依据 issue #2610 增补，取代原"仅 `--name`"的 stop 选择面：推荐的
  一 GPU 一 daemon 流程使裸 `stop` 必然歧义，用户只能猜 label。）
- 可发现性是 CLI contract 的一部分：所有 `stop`/`env` 选择失败的错误信息必须指出
  下一步（`uni-cumps status` 列出 record 名称与 GPU）；用户文档必须给出
  status → start → train → stop 完整生命周期，以及 record 丢失后 orphan daemon 的
  手动恢复路径。（2026-11-16，issue #2610。）
- Daemon record 以 `(UID, host, name)` 为 scope，持久化 canonical GPU UUID、绝对
  pipe/log 路径、control PID 与 Linux `/proc/<pid>/stat` start-time。PID 单独不可作为
  stop authority。
- GPU selector 必须解析为物理 GPU UUID；MIG 显式拒绝。多 GPU selector 解析为
  `single_task_multi_gpu` 未来形状但当前 fail closed，`all` 解析为 `task_per_gpu`
  诊断形状但同样不能启动。
- 拓扑模型必须保留 `single_gpu`、`single_task_multi_gpu`、`task_per_gpu` 三个模式；
  当前只允许 `single_gpu` 执行。
- 训练路径继续通过 `training.cuda_process_sharing: mps` 显式请求，不引入 `auto`，
  不隐式选择 daemon，不修改父 shell 环境。CLI 可与训练独立使用。

## Stable Contracts

- CLI entrypoint：`pyproject.toml` 的 `uni-cumps = "unilab.training.cuda_mps_cli:main"`。
- 生命周期 owner：`src/unilab/training/cuda_mps_cli.py`。
- 训练 probe contract：`src/unilab/training/cuda_process_sharing.py` 保持不变。
- Shell completion：`src/unilab/cli_completion.py` 提供 `cuda`/`mps` 与子命令补全。
- 测试：`tests/training/test_cuda_mps_cli.py` 使用 fake nvidia-smi、control daemon、
  process table 与 record 目录覆盖权限、stale、并发和 fail-closed 场景。

## Alternatives Considered

- 继续要求用户手写 shell 生命周期：无法统一安全 stop authority 和 future topology。
- 由 trainer 自动启动/停止 daemon：维持 ADR-0013 拒绝理由，训练进程无法安全拥有
  shared host service，并被并发/多用户运行放大。
- 引入 `cuda_process_sharing: auto`：没有可靠 host policy，且会破坏显式请求与
  fail-closed 语义。

## Consequences

- CLI 可以启动/停止 daemon，但这是显式用户操作，不是训练请求的隐式副作用。
- 没有 record、record scope 不匹配、或 PID/start-time 不再匹配时不能 stop。
- stale record 会被 quarantine/清理，但不能仅凭 pipe 路径 attach 到 unmanaged daemon。
- `single_task_multi_gpu` 与 `task_per_gpu` 只是 contract 形状，不构成支持声明；
  相关启用继续由 discussion #2063、roadmap #1672 与后续 lease/DP gate 决策。
- 该 CLI 不修改 SM percentage、OS affinity 或 memory limit，也不声明 multi-node 支持。

## Evidence In Repo

- `pyproject.toml`
- `src/unilab/training/cuda_mps_cli.py`
- `src/unilab/training/cuda_process_sharing.py`
- `src/unilab/cli_completion.py`
- `tests/training/test_cuda_mps_cli.py`
- `tests/test_completion.py`
- `docs/sphinx/source/en/2-user_guide/1-training/7-tensor_runtime_production.md`
- `docs/sphinx/source/zh_CN/2-user_guide/1-training/7-tensor_runtime_production.md`

## Related Documents

- {doc}`ADR-0013 CUDA MPS Single-Rank Execution Sharing </adr/ADR-0013-cuda-mps-single-rank-execution-sharing>`
- Issue #2079
- Discussion #1800
- Discussion #2063
- Roadmap #1672
