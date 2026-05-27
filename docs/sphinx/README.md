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

需要 Python >= 3.10。建议用 `uv`:

```bash
cd docs/sphinx

# 1. 装文档依赖
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 装 UniLab 本身,以便 autodoc 能 import unilab
#    没装时 conf.py 会自动 fallback 到 "prose-only" 构建,跳过 API reference
uv pip install -e ../..

# 3. 一次性构建
make html
# 输出:build/html/index.html(根 picker)、build/html/en/index.html、build/html/zh_CN/index.html

# 4. live preview(推荐写文档时用)
make live
# 默认监听 127.0.0.1:8000

# 5. CI 等价的"严格构建"(warning → error)
make strict
```

只想快速预览散文页(跳过 autodoc):

```bash
UNILAB_DOCS_SKIP_AUTODOC=1 make html
```

## CI / 部署

CI 工作流在 `.github/workflows/docs.yml`:

- **PR**: 只 build,不部署。failed 阻塞 PR(只看 build job,deploy 不跑)
- **push to main**: build + 用 deploy key 把 `build/html`(含 `/`、`/en/`、`/zh_CN/`)推到
  `unilabsim/UniLab-doc` 的 `gh-pages` 分支
- **手动**: `workflow_dispatch` 也能触发

### Deploy key 配置(一次性 setup)

CI 用 SSH deploy key 跨仓推送,不需要 PAT:

```bash
# 1. 在本地生成专用 keypair(不要复用个人 SSH key)
ssh-keygen -t ed25519 -C "unilab-doc-deploy" -f /tmp/unilab_doc_deploy -N ""

# 2. UniLab-doc 仓 → Settings → Deploy keys → Add deploy key
#    Title: unilab-docs-ci
#    Key:   贴 /tmp/unilab_doc_deploy.pub
#    ✅ Allow write access

# 3. UniLab 仓(本仓) → Settings → Secrets and variables → Actions → New repository secret
#    Name:  UNILAB_DOC_DEPLOY_KEY
#    Value: 贴 /tmp/unilab_doc_deploy(private,完整内容含 BEGIN/END 行)

# 4. 删除本地副本
shred -u /tmp/unilab_doc_deploy /tmp/unilab_doc_deploy.pub
```

CI step 用的是 `peaceiris/actions-gh-pages@v4`,详见 `.github/workflows/docs.yml`。

### UniLab-doc 仓的 Pages 设置

`Settings → Pages`:
- **Source**: Deploy from a branch
- **Branch**: `gh-pages` / `/ (root)`

第一次 CI 跑完会自动建 `gh-pages` 分支。

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
