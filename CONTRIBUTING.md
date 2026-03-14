# Contributing to UniLab

## 开发环境设置

1. Fork 并克隆仓库
2. 安装依赖：`uv sync --extra dev`
3. 创建功能分支：`git checkout -b feat/your-feature`

## 开发规范

- **Always use `uv run`**，不要直接使用 `python`
- 遵循代码风格：运行 `uv run ruff format .`
- 提交前检查：`uv run ruff check .`
- 运行测试：`uv run pytest`

## 提交规范

使用 Conventional Commits：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具链

## Pull Request 流程

1. 确保所有测试通过
2. 更新相关文档
3. 提交 PR 到 `main` 分支
4. 等待 code review

## 问题反馈

使用 GitHub Issues 报告 bug 或提出功能建议。
