# UniLab Sphinx Documentation

UniLab 的文档源码全部放在这个目录下。CI 在 push to `main` 时构建,然后通过 deploy key
推到独立仓库 [`unilabsim/UniLab-doc`](https://github.com/unilabsim/UniLab-doc) 的
`gh-pages` 分支,由 GitHub Pages 发布到
<https://unilabsim.github.io/UniLab-doc/>。

```
docs/sphinx/
├── Makefile               一次性 / live build / strict build 入口
├── requirements.txt       构建依赖
└── source/
    ├── conf.py            Sphinx 配置(theme、autodoc、intersphinx)
    ├── index.md           Landing
    ├── glossary.md
    ├── changelog.md
    ├── _static/           CSS / 图片 / 视频 / teaser 资源
    ├── _templates/        autosummary 模板
    ├── user_guide/        英文用户文档
    │   ├── getting_started/
    │   ├── backends/
    │   ├── algorithms/
    │   ├── tasks/
    │   ├── domain_randomization/
    │   ├── terrain/
    │   ├── manipulation/
    │   ├── tooling/
    │   └── zh_CN/         中文用户文档(从原 docs/users/zh_CN 迁来)
    ├── developer_guide/   英文 developer 文档
    │   ├── architecture/
    │   ├── contracts/
    │   ├── extending/
    │   ├── adr/           ADR(中英共用,使用英文文件名)
    │   └── zh_CN/         中文 developer 文档(从原 docs/developers/zh_CN 迁来)
    ├── agents/zh_CN/      Agent 速查
    ├── transfer/          Sim-to-real / sim-to-sim / framework migration
    └── api_reference/     autodoc 驱动的 Python API
```

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
# 输出:build/html/index.html

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

- **PR**: 只 build,不部署。失败会阻塞 PR(只看 build job,deploy 不跑)。
- **push to main**: build + 用 deploy key 把 `build/html` 推到
  `unilabsim/UniLab-doc` 的 `gh-pages` 分支。
- **手动**: `workflow_dispatch` 也能触发。

### Deploy key 配置(一次性 setup,已做过可跳过)

CI 用 SSH deploy key 跨仓推送,不需要 PAT。流程:

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

`Settings → Pages` 要选:

- **Source**: Deploy from a branch
- **Branch**: `gh-pages` / `/ (root)`

第一次 CI 跑完会自动建 `gh-pages` 分支。

## 写新文档的约定

1. **中英分层**:英文页面放在主结构(`user_guide/`, `developer_guide/` 等);中文页面放在对应的 `zh_CN/` 子目录,文件名与英文对齐。
2. **改 API 用 autodoc**:`api_reference/` 下页面只放 `automodule` / `autosummary` 指令,不要手写 API 表格——会和 docstring 漂移。
3. **链接源码**:正文里引用代码用 `{file}` / `{doc}` 角色或绝对 GitHub 链接,不要写相对路径(部署后路径会变)。
4. **图片放 `_static/`**:`![alt](../../_static/assets/foo.png)` 这种相对路径在 Sphinx 下能解析。
5. **ADR**:新 ADR 文件用 `ADR-NNNN-kebab-title.md` 命名,记得追加到 `developer_guide/index.md` 的 ADR toctree。

## 已知不全的部分

下列页面还是占位结构,内容待填:

- `developer_guide/architecture/*` (5 篇,主仓中文等价物在 `zh_CN/`)
- `developer_guide/contracts/*` (5 篇,主仓中文等价物在 `zh_CN/`)
- `developer_guide/extending/*` (4 篇)
- `user_guide/tasks/*` 个别页面
- `transfer/sim_to_real/*` 个别页面

中文版多数已有内容,可以作为英文翻译的底稿。
