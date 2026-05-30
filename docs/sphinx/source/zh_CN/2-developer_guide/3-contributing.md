# Contributing to UniLab

语言: 简体中文

## Development Environment Setup

1. Fork 并克隆仓库。
2. 按平台安装依赖:
   - macOS（MPS，安装 PyPI 的 torch wheel）: `uv sync`
   - Linux 默认（安装 PyTorch 官方 cu128 wheel；需要当前 PyTorch cu128 wheel 所支持的 NVIDIA 显卡与驱动栈）: `uv sync`
   - Linux AMD / ROCm 工作站: `make sync-rocm`，运行命令时使用 `uv run ...`
   - 从 ROCm 切回默认 CUDA / macOS profile: `git restore -- pyproject.toml uv.lock` 后重新 `uv sync --extra motrix`
   - 需要 Motrix 时，在命令后追加 `--extra motrix`
3. 创建分支，例如 `git checkout -b docs/improve-readme` 或 `git checkout -b fix/backend-bug`。

## Development Rules

- 始终使用 `uv run`；不要在 `uv run` 之外直接调用 `python`
- 代码相关提交前必须运行 `make check`
- 备份文件、临时导出物和历史兼容副本不要进入源码树；不要提交 `*.bak`、`*.tmp`、`*.old`、`*.orig` 或以 `~` 结尾的编辑器备份文件
- 只要改动用户可见工作流，就要同步维护顶层 `README.md`、`CONTRIBUTING.md`，以及 `docs/sphinx/source/en/`、`docs/sphinx/source/zh_CN/` 下对应文档
- 不要再往 `src/unilab/utils/` 塞新的 owner 逻辑；当前 `src/unilab/utils/*.py` 仅是过渡期 shim，计划在 `0.2.0` 删除
- 新模块/包名应直接表达 owner 职责：默认使用单数名词；只有在语义本身就是集合契约时才使用复数；工厂模块使用 `_factory` 后缀

## Read Before You Start

- 改训练入口、runner、env contract 或 backend 路径前，先看 [RL Infrastructure Development Standard](1-development-standard.md)
- 改协作流程或 issue / milestone 规则前，先看 [协作流程](2-collaboration.md)

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make sync-rocm      # Linux AMD / ROCm >= 7.1: sync deps and install torch==2.11.0+rocm7.2
make type           # mypy src/unilab + pyright
make check          # format + type（代码相关提交前必跑）
make test           # 非 slow 测试
make test-cov       # 非 slow 测试 + 覆盖率报告
make test-slow      # slow 集成测试和训练冒烟测试
make test-all       # make check && make test-cov
```

## Documentation Builds

文档 PR 的 GitHub Actions 只跑 prose-only HTML 构建: 设置
`UNILAB_DOCS_SKIP_AUTODOC=1`,不执行 `pip install -e .`,不生成 API reference,
也不发布外部文档仓库。开发者本地快速验证文档改动时运行:

```bash
uv run pytest tests/scripts/test_check_docs.py -q
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-build -b html -n source build/html
```

最终刷新完整站点、包含 API reference 并用于 `UniLab-doc` 发布流程时,在本地同步
UniLab 环境后使用 Sphinx 多进程构建:

```bash
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
```

## Commit Conventions

使用 Conventional Commits:

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 仅格式化，不改逻辑
- `refactor:` 代码重构
- `test:` 测试相关改动
- `chore:` 构建或工具链

## Testing

### Test Layout

```text
tests/
├── base/         # registry、backend 选择、env contract
├── config/       # Hydra / dataclass / reward 注入
├── envs/         # 环境配置与实例化
├── ipc/          # shared-memory 和 async-runner 原语
├── scripts/      # 训练脚本配置与入口工具
├── algos/        # runner 集成、RSL-RL PPO、MLX PPO
├── integration/  # 跨模块 reward / config 集成
└── utils/        # 辅助工具与实验跟踪
```

### Test Markers

- 普通测试（无标记）: 快速 unit / contract / env smoke，使用 `make test`
- `@pytest.mark.slow`: 完整训练/脚本冒烟，或累计耗时明显更高的 backend matrix，CI 会跳过，本地用 `make test-slow`
- macOS only: `test_mlx_ppo.py` 使用 `pytest.importorskip("mlx")`，在非 macOS 平台自动跳过

### Test Writing Principles

1. IPC、纯计算逻辑、快速 env/backend contract: 放在对应目录，不加 `slow`
2. 完整训练迭代、训练脚本启动、或累计成本高的 backend matrix: 加 `@pytest.mark.slow`
3. 训练脚本冒烟测试: 放在 `tests/scripts/`，对可选依赖使用 `pytest.importorskip`
4. 多进程测试使用 `_SPAWN_CTX = mp.get_context("spawn")`
5. 单进程 `SharedObsNormStats` 测试使用 `_ThreadingCtx`，因为 `multiprocessing.Queue.empty()` 在同进程内不可靠

### Running Tests

```bash
# 快速路径（与 CI 覆盖范围一致）
uv run pytest -m "not slow"

# 带覆盖率
uv run pytest -m "not slow" --cov=unilab --cov-report=term-missing

# 集成测试（需要 MuJoCo）
uv run pytest -m "slow" -v

# 完整训练冒烟测试
uv run pytest -m "slow" -v
```

## CI Workflow

指向 `main` 的 PR 会自动触发五个 job: `ruff-lint`、`ruff-format`、`mypy`、`pyright` 和 `test`。如果 PR 来自 fork，且 head branch 也叫 `main`，仓库会通过受限的 `pull_request_target` 通道来启动 CI，避免 GitHub Actions 继续静默不跑。workflow 也支持通过 `workflow_dispatch` 手动触发；文档改动会通过 pytest 套件参与校验，并且会自动取消同一 PR 分支上较早的进行中运行。

覆盖率策略: 默认 CI lane 会对非 slow 测试施加最低覆盖率门槛，这个门槛只应随着测试护栏增强而逐步上调，不应回退。

| Job | 内容 | 失败是否阻断 |
|-----|------|--------------|
| `ruff-lint` | 在 `ubuntu-slim` 上执行 `uv sync --only-group dev` + `uv run --no-sync ruff check --output-format=github .` | ✅ |
| `ruff-format` | 在 `ubuntu-slim` 上执行 `uv sync --only-group dev` + `uv run --no-sync ruff format --check .` | ✅ |
| `mypy` | 在 `ubuntu-slim` 上执行 `uv sync` + `uv run mypy src/unilab` | ✅ |
| `pyright` | 在 `ubuntu-slim` 上执行 `uv sync` + `uv run pyright` | ✅ |
| `test` | 在 `ubuntu-slim` 上以 Python 3.11 执行 `uv sync --extra motrix`，跳过 CUDA torch / nvidia wheel，安装 CPU torch，并用 `uv run --no-sync pytest -m "not slow" --cov=unilab --cov-report markdown-append:$GITHUB_STEP_SUMMARY --cov-fail-under=25` | ✅ |

只有协作元信息改动，例如 `LICENSE`、issue templates、`CODEOWNERS` 和 `.github/pull_request_template.md`，才会跳过 CI。文档改动会触发 CI，并由 `tests/scripts/test_check_docs.py` 校验。`Docs` workflow 是独立的 prose-only Sphinx 校验，支持 PR / push 触发，也可以在 GitHub Actions 网页端通过 `workflow_dispatch` 手动触发。

## Documentation Expectations

- 文档里的每条命令都必须能在当前仓库里对应到真实脚本、配置或 Makefile 目标
- 描述 backend 支持时，优先使用 `Registered`、`Configured`、`Benchmarked`、`Recommended`
- 使用相对链接，保证 GitHub 渲染正确
- 修改用户可见文档时，保持 `README.md`、`CONTRIBUTING.md` 和 `docs/sphinx/source/en/`、`docs/sphinx/source/zh_CN/` 下对应文档在结构上对齐
- 如果提到 CI、日志目录或支持矩阵，请对照 `.github/workflows/ci.yml`、`scripts/` 和 `conf/` 再核对一次

## GitHub Collaboration Model

- **Issue**: 一个 issue 对应一个可执行工作项
- **Milestone**: 阶段性目标，例如 `M1`
- **PR**: 必须链接 driving issue，并列出验证命令和影响范围
- **CODEOWNERS**: 表达 review ownership，不表示执行 ownership

更多协作约定见 [协作流程](2-collaboration.md)。

## Pull Request Workflow

1. 对代码或配置改动，先在本地运行 `make check`，确保 lint、mypy 和 pyright 通过。
2. 对代码改动，再在本地运行 `make test`，确保非 slow 测试通过。
3. 如果改到了 IPC、Runner 或 Config，补充或更新对应测试。
4. 对 docs-only 改动，至少运行 `uv run pytest tests/scripts/test_check_docs.py -q`。
5. 如果改动了仓库 hygiene 规则，运行 `uv run pytest tests/scripts/test_repo_hygiene.py -q`。
6. 链接对应 GitHub issue，并在 PR 模板中填写验证和影响范围。
7. 向 `main` 分支发起 PR，并等待 CI 全绿。
8. 等待 code review。

## Issue Reports

使用 GitHub Issues 报告 bug 或提出功能建议。

## Configuration System

UniLab 使用 Hydra + dataclass 配置系统:

- **添加新的 task 配置入口**: 在 `conf/{algo}/task/...` 下创建单一 owner YAML，并使用 `# @package _global_`
- **修改超参数**: 编辑对应 YAML，或使用 `algo.num_envs=2048` 这样的 CLI override
- **添加新算法**: 在 `structured_configs.py` 中添加 dataclass，并创建对应的 `conf/` 目录

更多细节见 {doc}`训练指南 </zh_CN/1-user_guide/3-training>` 的 Hydra 部分，以及 {doc}`Development Standard <1-development-standard>`。

## Navigation

- Index: [Documentation](../0-index.md)
- Previous: [Collaboration](2-collaboration.md)
