# docs/sphinx Agent Guide

这个文件规范 agent 在 `docs/sphinx/` 下撰写和修改文档内容的行为。仓库顶层
`AGENTS.md` / `CLAUDE.md` 仍然优先适用于代码、配置、测试和 PR 流程；本文件只补充
Sphinx 文档写作规则。

## Ground Truth

- 文档基础设施、构建和部署：`docs/sphinx/README.md`
- Sphinx 配置：`docs/sphinx/source/conf.py`
- 架构标准：`docs/sphinx/source/zh_CN/developer_guide/development-standard.md`
- ADR：`docs/sphinx/source/adr/ADR-0000-index.md`
- 术语表：`docs/sphinx/source/glossary.md`
- 文档检查：`tests/scripts/doc_checks.py`、`tests/scripts/test_check_docs.py`

## Source Structure

```text
source/
├── index.md                 # root redirect → en/index.html (no language picker)
├── en/                      # English tree (site root, shown in sidebar nav)
├── zh_CN/                   # Chinese tree (hidden from sidebar, via switcher only)
├── adr/                     # shared ADR, one set
├── api_reference/           # shared autodoc output from src/ docstrings
├── glossary.md
├── changelog.md
├── _static/
└── _templates/
    ├── sidebar/
    │   └── lang_switcher.html   # language dropdown in sidebar
    └── autosummary/
```

`source/en/` is the **default site root**. Visitors land on `/en/index.html`
directly (root `index.md` is a redirect). The sidebar navigation tree only
shows English pages. `source/zh_CN/` pages are built but accessible only
through the sidebar language switcher — they never appear in the navigation
tree.

`source/en/` and `source/zh_CN/` are parallel language roots, but they are
**not currently a strict 1:1 path mirror**. The Chinese user guide still keeps
compatibility paths such as `01-getting-started.md`, `A-getting-started/`, and
`C-algorithms/`. Do not rename those paths just to mirror English during an
unrelated documentation change. The language switcher uses an explicit path map
in `conf.py` (`_LANGUAGE_PATH_FORWARD`) to handle the mismatch.

Do not add per-page language button blocks or hand-written cross-language
navigation. The sidebar language switcher handles language changes globally.

## Core Principles

1. **Evidence only**: only document facts that can be verified in `src/`,
   `conf/`, `tests/`, `scripts/`, ADRs, or generated support data.
2. **Code is the source of truth**: names, signatures, defaults, Hydra keys, and
   commands follow the repository, not memory.
3. **Owner layer first**: scripts assemble; contracts live in backend, env,
   registry/config, runner/IPC, or algorithm owner layers.
4. **Link, do not duplicate**: English developer pages should summarize and link
   to ADRs or the development standard instead of copying the full Chinese
   standard.
5. **Config first**: backend/task/reward behavior belongs in Hydra owner YAMLs
   and registries where possible.
6. **API reference is autodoc**: `source/api_reference/` pages should contain
   autodoc/autosummary directives. Improve API prose in `src/unilab/**/*.py`
   docstrings.
7. **Use canonical commands**: examples use `uv run scripts/...`, not
   `python scripts/...`, `uv run python scripts/...`, or removed `unilab train`
   style commands.

## Before Writing

1. Decide the language root: `source/en/` or `source/zh_CN/`.
2. Locate the topic in `user_guide`, `developer_guide`, `transfer`, or `agents`.
3. Search first with `rg` / `rg --files`; update an existing page instead of
   creating a duplicate.
4. Gather evidence near the claim:
   - algorithms and tasks: `conf/`, `scripts/train_*.py`, `src/unilab/algos/`
   - env contract: `src/unilab/base/np_env.py`, `src/unilab/training/rsl_rl.py`
   - backend contract: `src/unilab/base/backend/base.py`
   - registry: `src/unilab/base/registry.py`
   - runner/IPC: `src/unilab/ipc/`, `src/unilab/training/run.py`
   - architecture: ADRs and `development-standard.md`
5. Check the other language for a related topic, but do not force path mirroring
   or mass renames.
6. If adding or deleting a page, update the relevant toctree.

## Navigation Rules

- **English pages**: do not hand-write `## Navigation`, `Previous`, or `Next`
  sections. Furo/Sphinx prev-next navigation comes from toctree order. If an
  English page in your edit scope still has a manual navigation block, remove it.
- **Chinese non-index pages**: keep the current checked shape: language line plus
  `## Navigation` with an Index link. The test contract is still enforced by
  `check_zh_cn_doc_shape`.
- **Root/language switching**: route language changes through the root landing and
  language switcher. Do not add ordinary-page language button pickers.

## Link And Path Rules

| Scenario | Use |
| --- | --- |
| Same-language doc link | MyST `{doc}` relative path, such as `{doc}`algorithms/ppo`` |
| Shared ADR or root-level page | Absolute `{doc}` path, such as `{doc}`/adr/ADR-0003-task-owner-and-config-compose-contract`` |
| Cross-language canonical link | Absolute `{doc}` path, such as `{doc}`/zh_CN/developer_guide/development-standard`` |
| Source/config/test path in prose | Backticks: `src/unilab/base/np_env.py` |
| GitHub source link | Full GitHub URL to an existing file when a clickable source link is needed |

Avoid old paths such as `docs/users/`, `docs/developers/`, and
`../../README.md` from Sphinx source pages.

## zh_CN Shape

Every `source/zh_CN/**.md` non-index page must keep this shape:

```markdown
# <标题>

语言: 简体中文

<正文>

## Navigation

- Index: [Documentation](../index.md)
```

The relative depth of the Index link may vary. Do not paste English prose into a
Chinese placeholder; a short honest Chinese placeholder is better than a fake
translation.

## ADR Rules

ADR files live in `source/adr/` and are shared across languages.

- New ADR names use `ADR-NNNN-kebab-title.md`.
- Use `source/adr/ADR-TEMPLATE.md`.
- Required fields: `- Status:`, `- Date:`, `- Owners:`, `- Supersedes:`,
  `- Superseded by:`, `## Alternatives Considered`, `## Evidence In Repo`,
  `## Related Documents`.
- Link ADRs from developer docs with absolute `{doc}` paths.
- ADR text is currently Chinese-first to avoid governance drift; English pages
  can summarize and link.

## Support Matrix

`source/zh_CN/user_guide/E-reference/01-backend-support-matrix.md` has a
generated block owned by `scripts/generate_support_matrix.py`.

After task/backend support changes, run:

```bash
uv run scripts/generate_support_matrix.py --write
```

Do not hand-edit the generated block.

## Toctree Rules

- Add pages to the relevant language index or section toctree.
- English toctrees should point to English pages or shared pages.
- Chinese toctrees may keep legacy numbered paths for compatibility.
- Deleting or renaming a page requires `rg` for all `{doc}` and toctree
  references. Avoid renames unless the migration is explicitly in scope.

## Concrete Anti-Patterns

- Marketing claims without benchmark evidence.
- "Coming soon", roadmap promises, or unsupported feature claims.
- Backend support claims that are not backed by registry/config/test data.
- `training.sim_backend=...` as a standalone backend switch. Use
  `task=<task>/<backend>` or `task=<algo>/<task>/<backend>`.
- Env hot paths that parse assets/XML or probe backend private methods.
- Hand-written API signatures in `source/api_reference/`.
- English `## Navigation` blocks.

## Validation

For docs-only changes, run:

```bash
uv run pytest tests/scripts/test_check_docs.py -q
cd docs/sphinx
UNILAB_DOCS_SKIP_AUTODOC=1 uv run sphinx-build -b html -n source build/html
```

If you changed generated support data, run the generator first. If you changed
`src/unilab/**/*.py` or training behavior, follow the top-level repo validation
rules instead of treating the change as docs-only.

Before reporting success, confirm:

- all cited `src/`, `conf/`, `tests/`, and `scripts/` paths exist;
- English pages in scope have no manual navigation block;
- root, `en/index.md`, and `zh_CN/index.md` remain included in toctrees;
- any warnings from Sphinx are understood and not introduced by the current edit.
