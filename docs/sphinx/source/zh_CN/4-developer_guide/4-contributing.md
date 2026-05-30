# 为 UniLab 做贡献

语言: 简体中文

本页概述面向贡献者的仓库工作流。契约与架构细节见
{doc}`1-architecture/1-overview`。

## 环境

```bash
uv sync
uv sync --extra motrix
make sync-rocm
make sync-xpu
```

请使用 `uv run` 运行命令。不要在 `uv run` 之外直接调用 `python`。

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

## Navigation

- Index: [文档](0-index.md)
