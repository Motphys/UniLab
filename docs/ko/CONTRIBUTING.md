# Contributing to UniLab

언어: [English](../../CONTRIBUTING.md) | [简体中文](../zh_CN/CONTRIBUTING.md) | [日本語](../ja/CONTRIBUTING.md) | 한국어

## Development Environment Setup

1. 저장소를 fork한 뒤 clone합니다.
2. 사용하는 플랫폼에 맞게 의존성을 설치합니다:
   - macOS (MPS): `uv sync`
   - Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8): `uv sync --extra cu124`
   - Motrix가 필요하면 `--extra motrix`를 추가합니다
3. `git checkout -b docs/improve-readme` 또는 `git checkout -b fix/backend-bug` 같은 branch를 만듭니다.

## Development Rules

- 항상 `uv run`을 사용하고, `uv run` 밖에서 `python`을 직접 호출하지 마세요
- 코드 관련 commit 전에는 `make check`를 실행하세요
- 사용자에게 보이는 workflow를 바꿨다면 top-level `README.md`, `CONTRIBUTING.md`, 그리고 `docs/{en,zh_CN,ja,ko}/` 아래 대응되는 각 언어 문서를 함께 맞춰야 합니다

## Read Before You Start

- 학습 엔트리포인트, runner, env contract, backend path를 바꾸기 전에 [RL Infrastructure Development Standard](00-development-architecture.md)를 읽으세요
- 협업 흐름이나 issue / milestone 규칙을 바꾸기 전에 [06-collaboration.md](06-collaboration.md)를 읽으세요

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make type           # mypy src/unilab + pyright
make check          # format + type (코드 관련 commit 전에 필수)
make test           # non-slow 테스트
make test-cov       # non-slow 테스트 + coverage report
make test-slow      # slow integration test (MuJoCo 필요)
make test-veryslow  # 전체 학습 smoke test (분 단위)
make test-all       # make check && make test-cov
```

## Commit Conventions

Conventional Commits를 사용합니다:

- `feat:` 새 기능
- `fix:` 버그 수정
- `docs:` 문서 업데이트
- `style:` 포맷만 변경, 로직 변경 없음
- `refactor:` 코드 리팩터링
- `test:` 테스트 관련 변경
- `chore:` 빌드 또는 툴링

## Testing

### Test Layout

```text
tests/
├── base/         # registry, backend 선택, env contract
├── config/       # Hydra / dataclass / reward injection
├── envs/         # 환경 설정과 인스턴스화
├── ipc/          # shared-memory 와 async-runner primitive
├── scripts/      # 학습 스크립트 설정과 엔트리포인트 도구
├── algos/        # runner 통합, RSL-RL PPO, MLX PPO
├── integration/  # 모듈 간 reward / config 통합
└── utils/        # 보조 유틸리티와 experiment tracking
```

### Test Markers

- 일반 테스트 (marker 없음): MuJoCo가 필요 없고 `make test`로 실행합니다
- `@pytest.mark.slow`: MuJoCo 환경이 필요하며 CI에서는 건너뛰고 로컬에서는 `make test-slow`로 실행합니다
- `@pytest.mark.veryslow`: 전체 학습 iteration 또는 스크립트 smoke test이며 `make test-veryslow`로 명시 실행합니다
- macOS only: `test_mlx_ppo.py`는 `pytest.importorskip("mlx")`를 사용하며 non-macOS 플랫폼에서 자동 skip됩니다

### Test Writing Principles

1. IPC 또는 순수 계산 로직: `tests/ipc/` 또는 해당 모듈 테스트 디렉터리에 두고 `slow`를 붙이지 않습니다
2. Runner 또는 실제 Env에 의존하는 테스트: `tests/algos/`에 두고 `@pytest.mark.slow`를 붙입니다
3. 학습 스크립트 smoke test: `tests/scripts/`에 두고 선택 의존성에는 `pytest.importorskip`을 사용합니다
4. multiprocessing 테스트에서는 `_SPAWN_CTX = mp.get_context("spawn")`를 사용합니다
5. 단일 process `SharedObsNormStats` 테스트에서는 `_ThreadingCtx`를 사용합니다. `multiprocessing.Queue.empty()`는 같은 process 안에서 신뢰할 수 없기 때문입니다

### Running Tests

```bash
# Fast path (CI와 동일한 범위)
uv run pytest -m "not slow and not veryslow"

# coverage 포함
uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report=term-missing

# integration test (MuJoCo 필요)
uv run pytest -m "slow and not veryslow" -v

# 전체 학습 smoke test
uv run pytest -m veryslow -v
```

## CI Workflow

`main` 대상 PR은 자동으로 세 개의 job을 실행합니다. 현재 workflow는 PR이 `main`에 merge된 뒤 같은 CI 세트를 다시 돌리지 않습니다.

| Job | 내용 | 실패 시 차단 여부 |
|-----|------|-------------------|
| `lint` | `ruff check` + `ruff format --check` | ✅ |
| `typecheck` | `mypy src/unilab` + `pyright` | ✅ |
| `test` | `pytest -m "not slow and not veryslow" --cov --cov-fail-under=10` | ✅ |

`*.md`, `docs/**`, issue templates, `CODEOWNERS` 같은 docs-only / 협업 메타데이터 변경은 CI를 트리거하지 않습니다.

## Documentation Expectations

- 문서에 적힌 모든 명령은 현재 저장소의 실제 script, config, Makefile target과 일치해야 합니다
- backend 지원을 설명할 때는 `Registered`, `Configured`, `Benchmarked`, `Recommended` 같은 표현을 우선 사용하세요
- GitHub에서 올바르게 렌더링되도록 상대 링크를 사용하세요
- 사용자 대상 문서를 바꾸면 English, zh_CN, Japanese, Korean 네 벌의 구조를 엄격히 맞추세요
- CI, 로그 루트, 지원 매트릭스를 언급할 때는 `.github/workflows/ci.yml`, `scripts/`, `conf/`와 대조하세요

## GitHub Collaboration Model

- **Issue**: 하나의 실행 가능한 작업 항목마다 하나의 issue
- **Milestone**: `M1` 같은 단계 목표
- **PR**: driving issue를 반드시 link하고 검증 명령과 영향 범위를 적어야 합니다
- **CODEOWNERS**: 실행 owner가 아니라 review owner를 나타냅니다

더 자세한 협업 규칙은 [06-collaboration.md](06-collaboration.md)를 보세요.

## Pull Request Workflow

1. 코드 또는 config를 바꿨다면 로컬에서 `make check`를 실행해 lint, mypy, pyright를 통과시키세요.
2. 코드 변경이 있다면 로컬에서 `make test`를 실행해 non-slow 테스트를 통과시키세요.
3. IPC, Runner, Config를 건드렸다면 대응되는 테스트를 추가하거나 갱신하세요.
4. docs-only 변경이라면 최소한 Markdown 링크, 파일 경로, script 이름, 명령 인자를 다시 확인하세요.
5. 관련 GitHub issue를 link하고 PR template에 검증 내용과 영향 범위를 적으세요.
6. `main`을 대상으로 PR을 열고 CI가 green이 될 때까지 기다리세요.
7. code review를 기다리세요.

## Issue Reports

버그 보고나 기능 제안은 GitHub Issues를 사용하세요.

## Configuration System

UniLab은 Hydra + dataclass 구성을 사용합니다:

- **새 task 추가**: `conf/{algo}/task/` 아래에 YAML을 만들고 `# @package _global_`를 사용합니다
- **하이퍼파라미터 변경**: 해당 YAML을 수정하거나 `algo.num_envs=2048` 같은 CLI override를 사용합니다
- **새 algorithm 추가**: `structured_configs.py`에 dataclass를 추가하고 해당 `conf/` 디렉터리를 만듭니다

자세한 내용은 [Training Guide](03-training.md)의 Hydra 섹션과 [Development Architecture](00-development-architecture.md)를 참고하세요.
