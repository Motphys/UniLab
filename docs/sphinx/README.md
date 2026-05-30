# UniLab Sphinx Documentation

UniLab 的文档源码全部在这里,**双语平行结构**——`source/en/` 是英文版,`source/zh_CN/`
是中文版,`adr/`、`api_reference/`、`glossary.md`、`changelog.md`、`_static/` 共享。

CI 在 push to `main` 时构建 Sphinx,通过 deploy key 推到独立仓
[`unilabsim/UniLab-doc`](https://github.com/unilabsim/UniLab-doc) 的 `gh-pages` 分支,
由 GitHub Pages 发布到 <https://unilabsim.github.io/UniLab-doc/>。

## 目录结构

```
docs/sphinx/
├── AGENTS.md                 ← agent 写文档时必读
├── README.md                 ← 本文件;build / deploy / 目录结构
├── Makefile                  ← html / live / strict 三档构建
├── requirements.txt          ← Sphinx + 扩展依赖,不含 unilab 本身
└── source/
    ├── conf.py               ← Sphinx 配置
    ├── index.md              ← 根 landing(语言 picker)
    │
    ├── _static/              ← 共享:CSS / 图片 / 视频 / teaser
    ├── _templates/           ← 共享:autosummary 模板
    ├── adr/                  ← 共享:ADR(中英共用,中文为主)
    ├── api_reference/        ← 共享:autodoc 驱动(英文,从 src/ docstring 拉)
    ├── glossary.md           ← 共享:术语表
    ├── changelog.md          ← 共享
    │
    ├── en/                   ← 英文版
    │   ├── index.md          ← 英文站根
    │   ├── user_guide/       ← getting_started / backends / algorithms / tasks / DR / terrain / manipulation / tooling
    │   ├── developer_guide/  ← architecture / contracts / extending / contributing
    │   ├── transfer/         ← sim_to_real / sim_to_sim / framework_migration
    │   └── agents/           ← 占位,待写
    │
    └── zh_CN/                ← 中文版
        ├── index.md          ← 中文站根
        ├── user_guide/       ← 01-getting-started / 02-simulation-backends / ... / A-E 速查
        ├── developer_guide/  ← development-standard / collaboration / domain-randomization-contract / ...
        ├── transfer/         ← 占位,待写
        └── agents/           ← Agent 速查(中文,01-agent-quick-reference.md)
```

URL 模式:`/`(语言 picker)、`/en/...`、`/zh_CN/...`。两种语言路径 1:1 镜像。

## 本地构建

需要 Python >= 3.10。建议用 `uv`。快速预览散文页时跳过 autodoc:

```bash
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-build -b html -n source build/html
```

最终发布前的完整 API 文档构建由开发者在本地执行。先在仓库根目录同步 UniLab
依赖并安装 Sphinx 依赖,再用 Sphinx 多进程构建:

```bash
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
```

需要本地 live preview 时:

```bash
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run --no-project --with-requirements requirements.txt sphinx-autobuild --watch ../../src source build/html -n
```

## CI / 部署

CI 工作流在 `.github/workflows/docs.yml`:

- **PR**: prose-only HTML build,跳过 API reference,failed 阻塞 PR
- **push to main**: prose-only HTML build,不部署
- **手动**: `workflow_dispatch` 可在 GitHub Actions 网页端触发同一套 prose-only CI

CI 明确设置 `UNILAB_DOCS_SKIP_AUTODOC=1`,不执行 `pip install -e .`,不安装 UniLab
运行时依赖,也不跑 linkcheck。完整构建和站点发布由开发者本地完成。

### UniLab-doc 仓的 Pages 设置

发布站点托管在 [`unilabsim/UniLab-doc`](https://github.com/unilabsim/UniLab-doc)。
该仓库的 Pages 设置保持:

- **Source**: Deploy from a branch
- **Branch**: `gh-pages` / `/ (root)`

### 本地发布到 UniLab-doc

完整 API 文档在本地构建完成后,把 `docs/sphinx/build/html/` 的内容同步到
`unilabsim/UniLab-doc` 的 `gh-pages` 分支根目录:

```bash
# 在 UniLab 仓库根目录
uv sync
uv pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
uv run --no-sync sphinx-build -j auto -b html -n source build/html
cd ../..
```

准备发布仓库。第一次发布时克隆:

```bash
git clone -b gh-pages git@github.com:unilabsim/UniLab-doc.git ../UniLab-doc
```

如果本地已经有 `../UniLab-doc`,先更新到最新:

```bash
git -C ../UniLab-doc checkout gh-pages
git -C ../UniLab-doc pull --ff-only
```

同步构建产物并推送:

```bash
rsync -a --delete --exclude .git docs/sphinx/build/html/ ../UniLab-doc/
touch ../UniLab-doc/.nojekyll

SOURCE_SHA=$(git rev-parse --short HEAD)
cd ../UniLab-doc
git status --short
git add -A
git commit -m "docs: publish UniLab@${SOURCE_SHA}"
git push origin gh-pages
```

`.nojekyll` 必须保留,否则 GitHub Pages 可能无法正确服务 Sphinx 的 `_static/`
等目录。推送后站点会更新到 <https://unilabsim.github.io/UniLab-doc/>。
不要把 `docs/sphinx/build/html/` 提交回 UniLab 主仓。

## 写新文档的约定

详细规则见 [`AGENTS.md`](AGENTS.md)。要点:

1. **双语平行**:英文进 `source/en/<section>/`,中文进 `source/zh_CN/<section>/`,**路径 1:1 镜像**(同名文件)
2. **共享内容**:ADR、API reference、glossary、changelog、`_static` 都在根目录共享,不写双语
3. **跨语言引用**:跨语言时用绝对路径 `/<lang>/<section>/<page>`,如 `{doc}`/zh_CN/user_guide/index``
4. **API 文档 = autodoc**:`api_reference/` 下页面只放 `automodule` / `autosummary`,不写手写表格
5. **图片放 `_static/`**:Sphinx 下用 `_static/assets/foo.png` 这种路径
6. **ADR 命名**:`ADR-NNNN-kebab-title.md`,挂到 `adr/README.md` 索引和各语言 developer_guide 的 ADR toctree

## 已知不全(框架已搭,内容待填)

- `source/en/user_guide/` 个别页面(中文等价物在 `zh_CN/user_guide/`)
- `source/en/developer_guide/architecture/`、`contracts/`、`extending/` 各页(中文等价物在 `zh_CN/developer_guide/`)
- `source/en/transfer/sim_to_real/` 个别页面
- `source/zh_CN/transfer/` 全部(等英文稳定后再翻)
- `source/en/agents/` 全部
- `source/en/agents/` 全部(中文版在 `source/zh_CN/agents/01-agent-quick-reference.md`)

中文版多数已有完整内容,可作为英文翻译底稿;反过来,英文 transfer 一节内容完整,可作为中文翻译底稿。
