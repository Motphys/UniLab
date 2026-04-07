# Contributing to UniLab

## 开发环境设置

1. Fork 并克隆仓库
2. 按平台安装依赖：
   - macOS (MPS): `uv sync --extra dev`
   - Linux (CUDA 12.4): `uv sync --extra dev --extra cu126`
   - 需要 Motrix 时，在上面的命令后追加 `--extra motrix`
3. 创建分支，例如：`git checkout -b docs/improve-readme`、`git checkout -b fix/backend-bug`

## 开发规范

- **Always use `uv run`**，不要直接使用 `python`
- 代码相关提交前必须通过 `make check`（ruff lint + mypy + pyright）
- 用户可见工作流改动时，同步更新 `README.md`、`docs/` 和 `CONTRIBUTING.md`

## 开发前先看

- 改训练入口、runner、env contract、backend 路径前，先看 [RL Infra Development Standard](docs/00-development-architecture.md)
- 改协作流程、issue / milestone 规则时，再看 [docs/06-collaboration.md](docs/06-collaboration.md)

## 常用命令

```bash
make format      # ruff format + ruff check --fix
make type        # mypy src/unilab + pyright
make check       # format + type（提交前必跑）
make test        # 非 slow 单元测试
make test-cov    # 非 slow 测试 + 覆盖率报告
make test-slow   # slow 集成测试（需要 MuJoCo）
make test-veryslow  # veryslow 训练冒烟测试（分钟级）
make test-all    # make check && make test-cov
```

## 提交规范

使用 Conventional Commits：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 格式化（不影响逻辑）
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具链

## 测试

### 测试目录

```
tests/
├── base/         # registry、backend 选择、env contract
├── config/       # Hydra / dataclass / reward 注入
├── envs/         # 环境配置与实例化
├── ipc/          # shared-memory / async runner 原语
├── scripts/      # 训练脚本配置与入口工具
├── algos/        # runner 集成、RSL-RL PPO、MLX PPO
├── integration/  # 跨模块 reward / config 集成
└── utils/        # 工具函数与实验跟踪
```

### 测试标记

- **普通测试**（无标记）：不依赖 MuJoCo，默认 `make test`
- **`@pytest.mark.slow`**：需要 MuJoCo 环境，CI 跳过，本地用 `make test-slow` 运行
- **`@pytest.mark.veryslow`**：完整训练迭代或脚本冒烟测试，显式用 `make test-veryslow`
- **macOS only**：`test_mlx_ppo.py` 用 `pytest.importorskip("mlx")` 在非 macOS 平台自动跳过

### 写测试的原则

1. IPC / 纯计算逻辑 → 放 `tests/ipc/` 或对应子目录，无需 slow 标记
2. 依赖 Runner / 真实 Env 的测试 → 放 `tests/algos/`，加 `@pytest.mark.slow`
3. 训练脚本冒烟测试 → 放 `tests/scripts/`，用 `pytest.importorskip` 跳过缺失依赖
4. 多进程测试用 `_SPAWN_CTX = mp.get_context("spawn")`
5. `SharedObsNormStats` 的单进程测试用 `_ThreadingCtx`（`multiprocessing.Queue.empty()` 在同进程内不可靠）

### 运行测试

```bash
# 快速（CI 同款）
uv run pytest -m "not slow and not veryslow"

# 带覆盖率
uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report=term-missing

# 集成测试（需 MuJoCo）
uv run pytest -m "slow and not veryslow" -v

# 完整训练冒烟
uv run pytest -m veryslow -v
```

## CI 流程

PR 到 `main` 时自动触发三个 job；当前 workflow 不在 `main` 合并后重复跑同一套 CI。

| Job | 内容 | 失败即阻断 |
|-----|------|-----------|
| `lint` | `ruff check` + `ruff format --check` | ✅ |
| `typecheck` | `mypy src/unilab` + `pyright` | ✅ |
| `test` | `pytest -m "not slow and not veryslow" --cov --cov-fail-under=10` | ✅ |

纯文档和协作元信息改动（如 `*.md`、`docs/**`、issue templates、`CODEOWNERS`）不触发 CI。

## 文档改动预期

- 文档命令必须能在当前仓库结构中找到对应脚本、配置或 Makefile 目标
- 如果文档描述了 backend 支持，请优先使用 `Registered` / `Configured` / `Benchmarked` / `Recommended` 语义
- 使用相对链接，确保 GitHub 渲染可直接跳转
- 如果文档描述了 CI、日志目录或支持矩阵，请对照 `.github/workflows/ci.yml`、`scripts/` 和 `conf/` 检查一次

## GitHub 协作方式

- **Issue**：一个可执行工作项一个 Issue
- **Milestone**：阶段目标，例如 `M1`
- **PR**：必须链接驱动 Issue，写清验证命令和影响范围
- **CODEOWNERS**：用于 review ownership，不等于执行 owner

更多约定见 [docs/06-collaboration.md](docs/06-collaboration.md)。

## Pull Request 流程

1. 对代码或配置改动，本地运行 `make check` 确保 lint/mypy/pyright 通过
2. 对代码改动，本地运行 `make test` 确保非 slow 测试通过
3. 若改动了 IPC / Runner / Config，补充或更新对应测试
4. 若为 docs-only 变更，至少重新检查 Markdown 链接、文件路径、脚本名和命令参数
5. 链接对应 GitHub Issue，并按 PR 模板填写验证与影响范围
6. 提交 PR 到 `main` 分支，等待 CI 全绿
7. 等待 code review

## 问题反馈

使用 GitHub Issues 报告 bug 或提出功能建议。

## 配置系统

UniLab 使用 Hydra + dataclass 配置系统：

- **添加新任务**：在 `conf/{algo}/task/` 创建 YAML，使用 `# @package _global_`
- **修改超参数**：编辑对应 YAML 或使用 CLI 覆盖（`algo.num_envs=2048`）
- **添加新算法**：在 `structured_configs.py` 添加 dataclass，创建对应 `conf/` 目录

详见 [Training Guide](docs/03-training.md) 的 Hydra 说明，以及 [Development Architecture](docs/00-development-architecture.md)。
