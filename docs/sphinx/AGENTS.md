# docs/sphinx Agent Guide

UniLab 的全部文档源码都在 `docs/sphinx/` 下,通过 Sphinx 构建,由 CI 推送到独立的
`unilabsim/UniLab-doc` 仓库的 `gh-pages` 分支发布。这个文件是 agent / 维护者
在改文档时的边界规约。**不是用户文档**——用户入口看 `docs/sphinx/source/index.md`。

## 角色边界(谁动什么)

| 角色 | 改什么 | 不改什么 |
|------|--------|---------|
| User 文档作者 | `source/user_guide/`、`source/transfer/`(任务、训练、算法、迁移) | conf.py、Makefile、CI workflow |
| Developer 文档作者 | `source/developer_guide/architecture/`、`contracts/`、`extending/`、`adr/` | API reference 页面(autodoc 生成) |
| API 注释作者 | `src/unilab/**/*.py` 里的 docstring | `source/api_reference/*` 手写(由 autodoc 拉) |
| Agent 维护者 | `source/agents/zh_CN/` | 其他子树 |
| Infra(本仓 maintainer) | `conf.py`、`requirements.txt`、`Makefile`、`.github/workflows/docs.yml` | 内容页 |

不确定时:**先看本文件的"高风险区域",再动手**。

## 目录结构与定位规则

```
docs/sphinx/
├── AGENTS.md                  ← 本文件;改文档前先读
├── README.md                  ← 本地 build、deploy key 配置
├── Makefile                   ← html / live / strict 三档构建
├── requirements.txt           ← Sphinx + 扩展依赖,不含 unilab 本身
└── source/
    ├── conf.py                ← Sphinx 配置(theme、autodoc、intersphinx)
    ├── index.md               ← 站点 landing,toctree 入口
    ├── glossary.md            ← 术语表(中文为主)
    ├── changelog.md
    ├── _static/               ← 主题 CSS、图片资源
    │   ├── css/custom.css
    │   ├── images/            ← logo / 截图
    │   └── assets/            ← teaser 等大图
    ├── _templates/            ← autosummary 模板
    │
    ├── user_guide/            ← 英文用户文档(主结构)
    │   ├── index.md
    │   ├── getting_started/   ← install / quickstart / training / overrides
    │   ├── backends/          ← MuJoCo / Motrix
    │   ├── algorithms/        ← PPO / APPO / SAC / TD3 / FlashSAC / MLX-PPO 等
    │   ├── tasks/             ← G1 / Go2 / Allegro / Sharpa
    │   ├── domain_randomization/
    │   ├── terrain/
    │   ├── manipulation/
    │   ├── tooling/           ← W&B / ONNX export / scene export
    │   └── zh_CN/             ← 中文用户文档(从原 docs/users/zh_CN 迁来)
    │       ├── 01..05-*.md    ← 主路径
    │       ├── A-getting-started/
    │       ├── B-training/
    │       ├── C-algorithms/
    │       ├── D-tasks/
    │       └── E-reference/   ← 支持矩阵等
    │
    ├── developer_guide/       ← 英文 developer 文档
    │   ├── index.md
    │   ├── contributing.md
    │   ├── contributing_workflow.md
    │   ├── architecture/      ← runtime model / layer boundaries / scene composition / dev standard
    │   ├── contracts/         ← env / backend / DR / runner lifecycle / task owner
    │   ├── extending/         ← new task / backend / algorithm / terrain
    │   ├── adr/               ← ADR-NNNN-*.md(中英共用,英文文件名)
    │   └── zh_CN/             ← 中文 developer 文档(从原 docs/developers/zh_CN 迁来)
    │       ├── development-standard.md
    │       ├── collaboration.md
    │       ├── domain-randomization-contract.md
    │       ├── scene-composition-design.md
    │       ├── motion-asset-migration.md
    │       ├── motrix-contact-sensor-notes.md
    │       └── CONTRIBUTING.md
    │
    ├── agents/zh_CN/          ← Agent 速查(中文)
    │
    ├── transfer/              ← Sim-to-real / sim-to-sim / framework migration
    │   ├── sim_to_real/       ← G1 / Go2 / Allegro / safety / latency / ONNX
    │   ├── sim_to_sim/        ← MuJoCo ↔ Motrix 切换、contact / friction 对齐
    │   └── framework_migration/ ← Isaac Lab / legged_gym / RSL-RL / skrl 迁移
    │
    └── api_reference/         ← autodoc 驱动(从 src/unilab 拉)
        ├── index.md
        ├── base/  envs/  algos/  backend/  training/  ipc/  dr/  ...
        └── top_level.md
```

### 内容归属对照表

| 想加哪类文档 | 该放哪 |
|-------------|--------|
| 新机器人 / 新任务的"怎么用" | `source/user_guide/tasks/<task>.md`,同时在 `source/user_guide/index.md` toctree 加入 |
| 新算法的训练命令、超参 | `source/user_guide/algorithms/<algo>.md` |
| 新 backend 的 user-facing 差异 | `source/user_guide/backends/<backend>.md` |
| 新 contract(env / backend / DR / runner) | `source/developer_guide/contracts/<name>.md` + 起一个 ADR |
| 架构决策 / 边界变更 | `source/developer_guide/adr/ADR-NNNN-<slug>.md`(NNNN 从 0006 起递增) |
| 已有架构的"原理 / 设计" | `source/developer_guide/architecture/<topic>.md` |
| 中文版用户文档 | `source/user_guide/zh_CN/<同英文路径>` |
| 中文版 developer 文档 | `source/developer_guide/zh_CN/<同英文路径>` |
| API 注释 / 类型修正 | **不写在 docs 里**——改 `src/unilab/**/*.py` 的 docstring,autodoc 会拉 |

## 高风险区域(改之前停一下)

| 区域 | 不可破坏的不变量 |
|------|----------------|
| `source/conf.py` | `html_baseurl` 必须保留 `https://unilabsim.github.io/UniLab-doc/`(决定 sitemap 链接);`autodoc_mock_imports` 是 CI 在没装 mlx/motrix 时也能 build 的关键,删了 CI 会炸;`autosummary_generate` 与 `_UNILAB_AVAILABLE` 联动,不要硬编码 True。 |
| `source/index.md` toctree | 顶层 toctree 是站点导航的 ground truth,加新页要同时挂进对应的 `:caption:` 块。删页前先 grep `{doc}` 引用,否则 strict build 会失败。 |
| ADR 文件结构 | `check_adr_shape`(`tests/scripts/doc_checks.py`)要求每个 ADR 含 `- Status:`、`- Date:`、`- Owners:`、`- Supersedes:`、`- Superseded by:`、`## Alternatives Considered`、`## Evidence In Repo`、`## Related Documents` 这 8 个字段。**例外**:`ADR-0000-index.md` 和 `adr/README.md` 是索引文件,不受此检查约束。 |
| zh_CN 文档头规约 | `check_zh_cn_doc_shape` 要求中文页第 3 行是 `语言: 简体中文`,且必须包含 `## Navigation` 段落和 `- Index: [Documentation](../../index.md)` 链接。改文件时别动这三行。 |
| User 文档体积 | `check_user_doc_architecture`:`source/user_guide/zh_CN/` 下单文件不超过 120 行(`E-reference/01-backend-support-matrix.md` 例外)。超了就拆,不要塞大而全页面。 |
| Backend 支持矩阵 | `source/user_guide/zh_CN/E-reference/01-backend-support-matrix.md` 的"生成块"由 `scripts/generate_support_matrix.py` 写;**不要手改**,跑脚本: `uv run scripts/generate_support_matrix.py --write`。 |
| autodoc 目标 | `source/api_reference/*` 下页面只放 `autosummary` / `automodule` 指令。手写表格、手写类签名 → 注定漂移,禁止。 |
| `_static/` 资源 | 大文件(>1MB)放 `assets/`,不要塞 `images/`(后者属于主题 CSS 引用)。新增前 `du -h` 看一眼。 |

## Sphinx 构建注意点

- **`make html`** 是日常默认,**不带 `-W`**,允许警告;**`make strict`** 才把 warning 当 error(用于 release 前 audit)。
- **`make live`** 会监听 `../../src`,代码改动也会触发文档重 build——便于 docstring 联调。
- **`UNILAB_DOCS_SKIP_AUTODOC=1`** 跳过 autodoc(适合本地只调散文页时用,build 时间从 ~30s 降到 ~5s)。
- **mock import**:在 `conf.py.autodoc_mock_imports` 里加新依赖前确认它真的是 optional 的;主流依赖(numpy / torch / gymnasium)不要 mock。

## CI / 部署边界

- **PR**:`.github/workflows/docs.yml` 在 PR 上只 build,不 deploy。build 失败会阻塞 PR 合并;warning 不阻塞。
- **main push**:build 成功后,把 `build/html` 通过 SSH deploy key 推到 `unilabsim/UniLab-doc` 的 `gh-pages` 分支。`UniLab-doc` 仓本身只是 Pages 部署目标,**不要在那里 PR 文档内容**。
- **deploy key**:`UNILAB_DOC_DEPLOY_KEY` secret 由仓库 owner 配,agent 不直接接触。

## PR Gate(文档改动)

文档改动的 PR 也要过 CI。pure docs 改动:

1. 跑 `uv run pytest tests/scripts/test_check_docs.py -q`(`make test-all` 的子集),确保 0 失败。
2. 跑 `cd docs/sphinx && uv pip install -r requirements.txt && sphinx-build -b html -n source build/html`,确认 exit 0(warning 不阻塞)。
3. **不需要**跑全套 `make test-all`,除非 PR 同时碰了 `src/unilab/**/*.py`。

## 不要做的事

- **不要新建 `docs/users/`、`docs/developers/`、`docs/agents/`**——这些路径已被废弃,新文档统一进 `docs/sphinx/source/`。
- **不要把内容写进 `docs/sphinx/source/api_reference/`**——它由 autodoc 生成,手写会被覆盖或者形成漂移源。
- **不要绕开 `conf.py.autodoc_mock_imports` 直接 import 重依赖**(mlx、motrixsim 等)——CI runner 上装不了,会让全站 build 失败。
- **不要直接编辑 `unilabsim/UniLab-doc` 仓**——它是 deploy target,所有改动都从本仓 build 出。
- **不要在 zh_CN 文件里写绝对路径回 `docs/...`**——文档已在 Sphinx 树内,用相对路径 `../../index.md` 或 `{doc}` 角色。
