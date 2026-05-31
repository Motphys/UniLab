# 为 UniLab 做贡献

本页概述面向贡献者的仓库工作流。契约与架构细节见
{doc}`1-architecture/1-overview`。

## 环境

按平台安装依赖：

- macOS（MPS，PyPI torch wheel）：`uv sync`
- Linux NVIDIA（PyTorch cu128 wheel）：`uv sync`
- Linux AMD / ROCm：`make sync-rocm`，随后用 `uv run ...` 运行命令。要切回默认
  CUDA / macOS profile，执行 `git restore -- pyproject.toml uv.lock` 后重新
  `uv sync`。
- Linux Intel XPU：`make sync-xpu`
- 需要 Motrix backend 时追加 `--extra motrix`，例如 `uv sync --extra motrix`。

```bash
uv sync
uv sync --extra motrix
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
- 当改动影响用户可见工作流时，保持 `README.md`、`CONTRIBUTING.md`，以及
  `docs/sphinx/source/en/` 与 `docs/sphinx/source/zh_CN/` 下对应页面同步。

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

对于仅涉及文档的改动，运行：

```bash
uv run pytest tests/scripts/test_check_docs.py -q
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-build -b html -n source build/html
```

`Docs` GitHub Actions workflow 会在匹配的 PR 和 push 上运行同样的 prose-only
构建，也可以在 GitHub Actions 网页界面通过 `workflow_dispatch` 手动触发。它不会用
`pip install -e .` 安装 UniLab，不生成 API reference 页面，也不发布外部文档仓库。

如果要在本地对完整站点（含面向 `UniLab-doc` 发布流程的 API reference 页面）做最终
刷新，请从已同步的开发环境用并行 Sphinx 构建：

```bash
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
```

## Commit 与 PR 预期

- 使用 Conventional Commits，例如 `feat:`、`fix:`、`docs:`、`refactor:`、
  `test:` 与 `chore:`。
- 在 PR 中关联驱动该工作的 issue。
- 列出实际运行过的验证命令。
- 说明行为在 MuJoCo、Motrix、macOS 或 Linux 之间是否存在差异。
- 对于代码/配置改动，在依赖顶层 smoke 命令之前，先运行最接近所改动契约的
  测试。

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
├── algos/        # runner 集成、RSL-RL PPO、MLX PPO
├── integration/  # 跨模块 reward / config 集成
├── training/     # 训练运行辅助
└── utils/        # 辅助工具与实验跟踪
```

标记与跳过：

- 无标记的测试是快速 unit / contract / env smoke，由 `make test` 运行。
- `@pytest.mark.slow` 标记完整训练/脚本 smoke 或累计成本高的 backend matrix。CI
  会跳过，本地用 `make test-slow`。`slow` 标记在 `pyproject.toml` 中注册。
- MLX PPO 测试（`tests/algos/test_mlx_ppo.py`）使用 `pytest.importorskip(...)`，
  在 MLX 不可用时自动跳过，实际上保持 macOS only。

## CI 工作流

指向 `main` 的 PR 会运行 `.github/workflows/ci.yml` 中的五个 job：`ruff-lint`、
`ruff-format`、`mypy`、`pyright` 与 `test`。每个都是必需检查，也可通过
`workflow_dispatch` 手动触发。同一分支上进行中的运行会被自动取消。

| Job | 内容 |
| --- | --- |
| `ruff-lint` | `uv run --no-sync ruff check --output-format=github .` |
| `ruff-format` | `uv run --no-sync ruff format --check .` |
| `mypy` | `uv run mypy src/unilab` |
| `pyright` | `uv run pyright` |
| `test` | `uv sync --extra motrix`（CPU torch），再 `uv run --no-sync pytest -m "not slow" --cov=unilab --cov-fail-under=25` |

`test` job 施加覆盖率门槛（`--cov-fail-under=25`）；这个下限只随测试护栏增强而
逐步上调。文档改动由同一套件中的 `tests/scripts/test_check_docs.py` 校验。独立的
`Docs` workflow 运行 prose-only 的 Sphinx 构建。

## 文档预期

- 命令必须指向已签入的脚本、包入口、Makefile target 或 config owner。
- 后端与任务的支持声明应当使用证据等级，例如
  `Registered`、`Configured`、`Tested`、`Benchmarked` 或 `Recommended`。
- 不要把 `training.sim_backend=<backend>` 描述为独立的后端切换方式。在
  面向用户的命令中使用 `--sim <backend>`，并在内部选择 owner YAML 路径。
- 让英文页面不含手写的导航块。

## 配置改动

任务、后端、reward 与算法的选择应当属于 Hydra owner YAML。当添加或改动
一条可运行路径时，更新 `conf/` 下相关的 owner config，并用 `tests/config/`
或 `tests/scripts/` 下的测试验证脚本组合。

参见 {doc}`2-contracts/3-task_owner` 与
{doc}`../2-user_guide/1-training/2-hydra_config`。
