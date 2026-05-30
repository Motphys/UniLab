# UniLab Documentation

UniLab 的全部文档源码统一在 [`docs/sphinx/`](sphinx/) 下,通过 Sphinx 构建,部署到
<https://unilabsim.github.io/UniLab-doc/>。

发布入口:

- Project page: <https://unilabsim.github.io>
- Paper: <https://arxiv.org/abs/2605.30313>
- Documentation: <https://unilabsim.github.io/UniLab-doc/>

文档采用**双语平行结构**:`source/en/` 是英文版,`source/zh_CN/` 是中文版,
`source/adr/`、`source/api_reference/`、`source/glossary.md`、`source/changelog.md` 在两种语言间共享。

## 入口

| 内容 | 路径 |
|------|------|
| 英文用户指南 | [`docs/sphinx/source/en/2-user_guide/`](sphinx/source/en/2-user_guide/) |
| 中文用户指南 | [`docs/sphinx/source/zh_CN/1-user_guide/`](sphinx/source/zh_CN/1-user_guide/) |
| 英文 Developer 指南 | [`docs/sphinx/source/en/4-developer_guide/`](sphinx/source/en/4-developer_guide/) |
| 中文 Developer 指南 | [`docs/sphinx/source/zh_CN/2-developer_guide/`](sphinx/source/zh_CN/2-developer_guide/) |
| 英文 Deployment(sim-to-real / sim-to-sim) | [`docs/sphinx/source/en/3-deployment/`](sphinx/source/en/3-deployment/) |
| 中文 Agent 速查 | [`docs/sphinx/source/zh_CN/3-agents/`](sphinx/source/zh_CN/3-agents/) |
| ADR(共享,中文为主) | [`docs/sphinx/source/adr/`](sphinx/source/adr/) |
| API Reference(autodoc,英文) | [`docs/sphinx/source/api_reference/`](sphinx/source/api_reference/) |
| 术语表 | [`docs/sphinx/source/glossary.md`](sphinx/source/glossary.md) |

## 本地构建

```bash
cd docs/sphinx
uv pip install -r requirements.txt
make html        # 一次性构建
make live        # sphinx-autobuild,自动 reload
```

详细构建与部署流程见 [`docs/sphinx/README.md`](sphinx/README.md)。
Agent 写文档时应遵守的规则见 [`docs/sphinx/AGENTS.md`](sphinx/AGENTS.md)。
