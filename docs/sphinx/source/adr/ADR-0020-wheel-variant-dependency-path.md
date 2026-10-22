---
orphan: true
---

# ADR-0020 Wheel Variant Dependency Path

语言: 简体中文

- Status: Proposed
- Date: 2026-10-22
- Owners: UniLab Maintainers
- Supersedes: None
- Superseded by: None

## Context

Torch 的 CPU、CUDA、ROCm 与 XPU 二进制构建使用同一个包名，但 PEP 508 依赖
metadata 没有加速器维度。仓库此前用第二套依赖配置文件和 lock 覆盖 tracked
`pyproject.toml` / `uv.lock` 来安装 ROCm 构建。这会让工作区变脏、要求人工同步
两条依赖线，并让 PyPI 下游用户无法用一次依赖解析得到正确的 Torch 构建。

WheelNext / PEP 817 variant metadata 已经提供单一解析记录 CPU、CUDA、ROCm 与
XPU wheel 的路径。`unisim-core==2.1.1` 发布 wheel 后，实验版 uv 也能在不使用
UniSim source override 的情况下完成完整 lock 与同步。维护者选择不等待稳定版
uv 默认支持，先以钉版实验构建落地单一 variant 路径。

后续审查发现，`uv-wheelnext 0.9.14` 的 universal lock 能记录 Torch variant
wheel，但不会把 variant-scoped runtime dependencies（例如 CUDA 的 NVIDIA
库与 Triton、ROCm 的 Triton、XPU 的 Triton）写入 lock。真实 MI300X 的基本
训练路径可以通过，但 CUDA/Docker 与编译路径不能据此宣称完整。因此本 ADR 和
迁移 PR 保持 draft；合并前等待 uv 正式发布包含 variant-scoped dependency
lock/install 支持的版本，并用同一 lock 重新生成与验证。

## Decision

1. UniLab 只维护一个 `pyproject.toml` 和一个 wheel-variant-aware `uv.lock`。
2. Torch 依赖下限为 `2.10`，不再通过 uv source 或专用 PyTorch index 固定某个
   CUDA/ROCm 构建。
3. 仓库钉版 `uv-wheelnext 0.9.14` 作为安装与解析器。源码 checkout、CI 与容器
   都通过 `scripts/tools/install_uv_wheelnext.sh` 获取该构建。
4. 加速器选择属于安装器 variant 解析，不属于训练后端配置。运行时仍由 Torch 的
   `torch.version.hip`、设备类型和 backend capability 决定行为；ROCm/HIP 上训练
   device 继续使用 `cuda` 拼写。
5. `unilab doctor` 保留 AMD/ROCm 混装、provider 探测与驱动可见性诊断；它不是
   依赖 profile 切换器。

## Stable Contracts

- 仓库不得重新引入第二套 ROCm/XPU 依赖 profile、lock 交换 target，或直接依赖
  ROCm Triton 包；Torch variant 和其运行时依赖由 variant metadata 解析。
- 变更 Torch 下限、variant-aware resolver 或 `unisim-core` 发布依赖时，必须同
  一 PR 更新单一 lock，并用钉版 uv 验证。
- CI 的项目依赖 job 必须在 stable uv 安装 Python 后覆盖为 WheelNext uv；CPU
  覆盖 wheel 只能作为测试环境成本优化，不得成为用户安装路径。
- 文档必须写明实验 uv 版本与稳定 uv 的限制；稳定生态可用后应替换钉版脚本，
  而不是恢复双 profile。

## Alternatives Considered

- 等待稳定版 uv 与 PyPI 默认 variant 支持：真实 MI300X 已验证实验 wheel 选择，
  且 `unisim-core` wheel 消除了 source 解析障碍，但 lock 中的 variant-scoped
  dependencies 缺失是合并阻断项，不能靠恢复双 profile 或手工依赖清单绕过。
- 保留双 profile 并增加同步 lint：拒绝。该方案仍覆盖 tracked 文件，无法服务
  PyPI 下游用户，也不能消除人工同步责任。

## Consequences

- 当前稳定 uv 不能消费该 lock；开发、CI 和文档必须显式使用钉版实验构建。
- 普通 PyPI 镜像替换默认 index 可能绕过 WheelNext variant metadata，在稳定
  生态集成前不能作为 variant 安装建议。
- Torch 解析线从 2.14 降到 variant index 覆盖的 2.10，需要保留该线的 CUDA
  graph / compile 兼容处理，并用真实 ROCm 训练冒烟验证。
- 将来稳定 uv 支持当前 schema 后，应删除实验安装器钉版并重新验证镜像与
  PyPI 路径，但单一 lock contract 不变。

## Evidence In Repo

- `pyproject.toml`
- `uv.lock`
- `scripts/tools/install_uv_wheelnext.sh`
- `.github/actions/wheelnext-uv/action.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/tensor-runtime-ci.yml`
- `Dockerfile`
- `src/unilab/diagnostics.py`
- `tests/scripts/test_wheel_variant_dependency.py`
- `src/unilab/rl/algos/common/compile.py`
- `docs/sphinx/source/en/1-getting_started/2-installation.md`
- `docs/sphinx/source/zh_CN/1-getting_started/2-installation.md`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`ADR-0016 RL Reintegration Into unilab.rl And Single Distribution </adr/ADR-0016-rl-reintegration-into-unilab-rl>`
- [Issue #2500](https://github.com/Motphys/UniLab/issues/2500)
