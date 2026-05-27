# UniLab Documentation

UniLab 的全部文档源码已统一搬到 [`docs/sphinx/`](sphinx/) 下,通过 Sphinx 构建,部署到
<https://unilabsim.github.io/UniLab-doc/>。

## 入口

| 角色 | 入口 |
|------|------|
| 用户(中文) | [`docs/sphinx/source/user_guide/zh_CN/`](sphinx/source/user_guide/zh_CN/) |
| 用户(英文) | [`docs/sphinx/source/user_guide/`](sphinx/source/user_guide/) |
| Developer(中文) | [`docs/sphinx/source/developer_guide/zh_CN/`](sphinx/source/developer_guide/zh_CN/) |
| Developer(英文) | [`docs/sphinx/source/developer_guide/`](sphinx/source/developer_guide/) |
| ADR | [`docs/sphinx/source/developer_guide/adr/`](sphinx/source/developer_guide/adr/) |
| Agent 速查 | [`docs/sphinx/source/agents/zh_CN/`](sphinx/source/agents/zh_CN/) |
| 术语表 | [`docs/sphinx/source/glossary.md`](sphinx/source/glossary.md) |

## 本地构建

```bash
cd docs/sphinx
uv pip install -r requirements.txt
make html        # 一次性构建
make live        # sphinx-autobuild,自动 reload
```

详细构建与部署流程见 [`docs/sphinx/README.md`](sphinx/README.md)。
