# 为 UniLab 做贡献

本页概述面向贡献者的仓库工作流。契约与架构细节见
{doc}`1-architecture/1-overview`。

## 环境

按平台安装依赖。setup target 也会安装仓库检查所需的可选仿真器 extra：

- macOS（MPS，PyPI torch wheel）：`make setup-motrix`（或 `uv sync --extra mujoco`）
- Linux NVIDIA（PyTorch cu128 wheel）：`make setup`
- Linux AMD / ROCm：`make sync-rocm`，随后用 `uv run ...` 运行命令。要切回默认
  CUDA / macOS profile，执行 `git restore -- pyproject.toml uv.lock` 后重新
  `make setup`。
- Linux Intel XPU：`make sync-xpu`
- 如果更喜欢直接使用 uv，完整默认环境为 `uv sync --extra mujoco --extra motrix`；
  单后端使用 `--extra mujoco` 或 `--extra motrix`。

```bash
# 选择一条核心安装路径：
make setup
# make setup-motrix
make sync-rocm
make sync-xpu
```

请使用 `uv run` 运行命令。不要在 `uv run` 之外直接调用 `python`。

## 开发规则

- 始终使用 `uv run`。任何代码相关提交前先运行 `make check`。
- 不要提交备份或临时文件：不要 `*.bak`、`*.tmp`、`*.old`、`*.orig`，也不要以
  `~` 结尾的编辑器备份文件。
- 不要往 `src/unilab/utils/` 塞新的 owner 逻辑。那里的模块是过渡期 shim；应把
  长期逻辑上移到对应 owner 层或 `src/unilab/base/`。
- 模块命名表达 owner 职责：默认使用单数名词；只有当语义本身就是集合契约时才用
  复数；工厂模块使用 `_factory` 后缀。
- 代码注释、公共 API docstring、内部实现说明、TODO/FIXME 与配置注释默认使用
  英文。中文说明放在 `docs/sphinx/source/zh_CN/` 下的中文文档中，不在源码注释或
  配置注释中重复本地化说明。
- 当改动影响用户可见工作流时，保持 `README.md`、`CONTRIBUTING.md`，以及
  `docs/sphinx/source/en/` 与 `docs/sphinx/source/zh_CN/` 下对应页面同步。

## 注释语言规范

- 公共 API docstring 必须用英文描述 contract、参数、返回值与边界条件。
- inline comment 用英文解释非显而易见的实现意图、owner 边界，或
  backend/env/config 不变量；过时注释应删除，而不是机械翻译。
- TODO/FIXME 使用英文，并尽量写明后续处理所属的 owner layer 或外部依赖。
- Hydra YAML 与示例配置注释使用英文，因为它们与源码 contract 一起 review。
- 已存在的中英文混用注释按小 PR 分批迁移。优先级为公共 contract、backend
  adapter、env contract、training runner、config schema 和高频测试，再处理低风险
  示例。

## 常用命令

```bash
make format
make type
make check
make test
make test-cov
make test-slow
make test-all
```

文档改动在 `make test-all` 之外运行以下针对性验证：

```bash
uv run pytest tests/scripts/test_check_docs.py -q
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-build -b html -n source build/html
```

`Docs` GitHub Actions workflow 会在 base 为 `main` 的匹配 PR 和 `main` push 上运行
同样的 prose-only 构建，也可以在 GitHub Actions 网页界面通过 `workflow_dispatch`
手动触发。它不会用
`pip install -e .` 安装 UniLab，不生成 API reference 页面，也不发布外部文档仓库。

如果要在本地对完整站点（含面向 `UniLab-doc` 发布流程的 API reference 页面）做最终
刷新，请从已同步的开发环境用并行 Sphinx 构建：

```bash
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
```

## 协作

使用 Conventional Commit 标题。Issue 范围、roadmap 分支、PR 证据、本地和远程 gate、ADR
与发布的唯一规则来源是 {doc}`5-contributing_workflow`。创建或更新 PR 前，在最终 head
运行贴近改动契约的检查和 `make test-all`。

## 测试

测试按 owner 区域分组，位于 `tests/`：

```text
tests/
├── base/         # registry、backend 选择、env contract
├── config/       # Hydra / dataclass / reward 注入
├── envs/         # 环境配置与实例化
├── dr/           # domain-randomization 类型与 manager
├── terrains/     # 地形生成器与场景 materialization
├── ipc/          # shared-memory 与 async-runner 原语
├── scripts/      # 训练脚本配置与入口工具
├── algos/        # runner 集成、RSL-RL PPO
├── integration/  # 跨模块 reward / config 集成
├── training/     # 训练运行辅助
└── utils/        # 辅助工具与实验跟踪
```

标记与跳过：

- 无标记的测试是快速 unit / contract / env smoke，由 `make test` 运行。
- `@pytest.mark.slow` 标记完整训练/脚本 smoke 或累计成本高的 backend matrix。CI
  会跳过，本地用 `make test-slow`。`slow` 标记在 `pyproject.toml` 中注册。

`make test-slow` 运行须知：

- **测试专用 env 注册走子进程钩子**。off-policy / APPO 等 runner 通过
  `multiprocessing.spawn` 起 collector 子进程，spawn 出来的解释器**不会执行**
  `tests/conftest.py`，所以 `DummyFlatTest` 等测试 env 不能只在 conftest
  完成注册。`tests/conftest.py` 会向 `UNILAB_EXTRA_REGISTRY_PACKAGES` 环境
  变量注入 `tests._test_registry`，`unilab.base.registry.ensure_registries`
  在子进程内读到后再次完成注册。新增测试 env 时，把模块加进
  `tests/_test_registry/__init__.py` 的 `__unilab_registry_modules__`。
- **Replay 内存预算**。off-policy host shared memory 只包含固定深度 ingress，随
  `num_envs` 与 transition width 增长，不随 `replay_buffer_n` 增长。完整 ring 按
  `replay_buffer_n` 驻留在 CUDA/MPS learner device。如果运行时看到形如
  `MemoryError: estimated shared-memory allocation … exceeds /dev/shm
  available …`，说明本机 shared memory 不够容纳默认 ingress。device ring 分配失败
  会另行报告所需与可用的 accelerator budget。

## 文档预期

- 命令必须指向已签入的脚本、包入口、Makefile target 或 config owner。
- 后端与任务的支持声明应当使用证据等级，例如
  `Registered`、`Configured`、`Tested`、`Benchmarked` 或 `Recommended`。
- 不要把 `training.sim_backend=<backend>` 描述为独立的后端切换方式。在
  面向用户的命令中使用 `--sim <backend>`，并在内部选择 owner YAML 路径。
- 让英文页面不含手写的导航块。

## 配置改动

任务、后端、reward 与算法的选择应当属于 Hydra owner YAML。当添加或改动
一条可运行路径时，更新 `src/unilab/conf/` 下相关的 owner config，并用 `tests/config/`
或 `tests/scripts/` 下的测试验证脚本组合。

参见 {doc}`2-contracts/3-task_owner` 与
{doc}`../2-user_guide/1-training/2-hydra_config`。
