# Contributing to UniLab

## 开发环境设置

1. Fork 并克隆仓库
2. 安装依赖：`uv sync --extra dev`
3. 创建功能分支：`git checkout -b feat/your-feature`

## 开发规范

- **Always use `uv run`**，不要直接使用 `python`
- 提交前必须通过 `make check`（ruff lint + mypy + pyright）
- 动态任务状态不要写进 `README.md` 或临时 markdown，统一用 GitHub Issues / Milestones 跟踪

## 常用命令

```bash
make format      # ruff format + ruff check --fix
make type        # mypy unilab + pyright
make check       # format + type（提交前必跑）
make test        # 非 slow 单元测试
make test-cov    # 非 slow 测试 + 覆盖率报告
make test-slow   # slow 集成测试（需要 MuJoCo）
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

### 测试结构

```
tests/
├── conftest.py                    # 共享 fixtures（含 DummyFlatEnv，无需 MuJoCo）
├── ipc/                           # IPC 原语单元测试
│   ├── test_replay_buffer.py
│   ├── test_shared_onpolicy_storage.py
│   ├── test_shared_weight_sync.py
│   ├── test_shared_obs_stats.py
│   └── test_async_runner.py
├── base/
│   └── test_registry.py
├── config/
│   └── test_locomotion_params.py
├── scripts/
│   └── test_train_scripts.py
└── algos/
    ├── test_appo_runner.py        # @pytest.mark.slow
    ├── test_offpolicy_runner.py   # @pytest.mark.slow
    └── test_mlx_ppo.py            # macOS only（MLX 后端）
```

### 测试标记

- **普通测试**（无标记）：不依赖 MuJoCo，CI 中自动运行
- **`@pytest.mark.slow`**：需要 MuJoCo 环境，CI 跳过，本地用 `make test-slow` 运行
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
uv run pytest -m "not slow"

# 带覆盖率
uv run pytest -m "not slow" --cov=unilab --cov-report=term-missing

# 集成测试（需 MuJoCo）
uv run pytest -m slow -v
```

## CI 流程

PR 到 `main` 时自动触发三个 job；合并后不会在 `main` 上重复跑同一套 CI。

| Job | 内容 | 失败即阻断 |
|-----|------|-----------|
| `lint` | `ruff check` + `ruff format --check` | ✅ |
| `typecheck` | `mypy unilab` + `pyright` | ✅ |
| `test` | `pytest -m "not slow and not veryslow" --cov --cov-fail-under=10` | ✅ |

纯文档和协作元信息改动（如 `docs/**`、issue templates、`CODEOWNERS`）不触发 CI。

## GitHub 协作方式

- **Issue**：一个可执行工作项一个 Issue，不要把 milestone 任务堆在文档里
- **Milestone**：阶段目标，例如 `M1`
- **PR**：必须链接驱动 Issue，写清验证命令和影响范围
- **CODEOWNERS**：用于 review ownership，不等于执行 owner

更多约定见 [docs/06-collaboration.md](docs/06-collaboration.md)。

## Pull Request 流程

1. 本地运行 `make check` 确保 lint/mypy/pyright 通过
2. 本地运行 `make test` 确保单元测试通过
3. 若改动了 IPC / Runner / Config，补充或更新对应测试
4. 链接对应 GitHub Issue，并按 PR 模板填写验证与影响范围
5. 提交 PR 到 `main` 分支，等待 CI 全绿
6. 等待 code review

## 问题反馈

使用 GitHub Issues 报告 bug 或提出功能建议；阶段计划请使用 GitHub Milestones，不要继续放在 `README` 或临时 markdown 中。

## 配置系统

UniLab 使用 Hydra + dataclass 配置系统：

- **添加新任务**：在 `conf/{algo}/task/` 创建 YAML，使用 `# @package _global_`
- **修改超参数**：编辑对应 YAML 或使用 CLI 覆盖（`algo.num_envs=2048`）
- **添加新算法**：在 `structured_configs.py` 添加 dataclass，创建对应 `conf/` 目录

详见 `AGENTS.md` 配置系统章节。
