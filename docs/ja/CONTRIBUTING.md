# Contributing to UniLab

言語: [English](../../CONTRIBUTING.md) | [简体中文](../zh_CN/CONTRIBUTING.md) | 日本語 | [한국어](../ko/CONTRIBUTING.md)

## Development Environment Setup

1. リポジトリを fork して clone します。
2. 利用するプラットフォームに応じて依存関係を入れます:
   - macOS (MPS): `uv sync --extra dev`
   - Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8): `uv sync --extra dev --extra cu124`
   - Motrix が必要な場合は `--extra motrix` を追加します
3. `git checkout -b docs/improve-readme` や `git checkout -b fix/backend-bug` のように branch を切ります。

## Development Rules

- 常に `uv run` を使い、`uv run` の外で `python` を直接呼ばないでください
- コード関連の commit 前に `make check` を実行してください
- ユーザー向け workflow を変更した場合は、トップレベルの `README.md`、`CONTRIBUTING.md` と、`docs/{en,zh_CN,ja,ko}/` 配下の対応する各言語文書を同期してください

## Read Before You Start

- 学習 entrypoint、runner、env contract、backend path を変更する前に [RL Infrastructure Development Standard](00-development-architecture.md) を読んでください
- 協業フローや issue / milestone ルールを変更する前に [06-collaboration.md](06-collaboration.md) を読んでください

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make type           # mypy src/unilab + pyright
make check          # format + type（コード関連の commit 前に必須）
make test           # non-slow テスト
make test-cov       # non-slow テスト + coverage report
make test-slow      # slow integration test（MuJoCo が必要）
make test-veryslow  # 完全な学習 smoke test（分単位）
make test-all       # make check && make test-cov
```

## Commit Conventions

Conventional Commits を使います:

- `feat:` 新機能
- `fix:` bug 修正
- `docs:` ドキュメント更新
- `style:` フォーマットのみ、ロジック変更なし
- `refactor:` リファクタリング
- `test:` テスト関連
- `chore:` build / tooling

## Testing

### Test Layout

```text
tests/
├── base/         # registry、backend 選択、env contract
├── config/       # Hydra / dataclass / reward injection
├── envs/         # 環境設定とインスタンス化
├── ipc/          # shared-memory と async-runner primitive
├── scripts/      # 学習スクリプト設定と entrypoint ツール
├── algos/        # runner 統合、RSL-RL PPO、MLX PPO
├── integration/  # モジュール横断の reward / config 統合
└── utils/        # 補助 utility と experiment tracking
```

### Test Markers

- 通常テスト（marker なし）: MuJoCo 不要、`make test` で実行
- `@pytest.mark.slow`: MuJoCo 環境が必要、CI では skip、ローカルでは `make test-slow`
- `@pytest.mark.veryslow`: 学習 1 iteration 全体やスクリプト smoke test、`make test-veryslow` で明示実行
- macOS only: `test_mlx_ppo.py` は `pytest.importorskip("mlx")` を使い、非 macOS では自動 skip

### Test Writing Principles

1. IPC や純粋な計算ロジック: `tests/ipc/` または対応する module の test に置き、`slow` は付けない
2. Runner や実 Env に依存するテスト: `tests/algos/` に置き、`@pytest.mark.slow` を付ける
3. 学習スクリプトの smoke test: `tests/scripts/` に置き、任意依存には `pytest.importorskip` を使う
4. multiprocessing を使うテストでは `_SPAWN_CTX = mp.get_context("spawn")` を使う
5. 単一 process の `SharedObsNormStats` テストでは `_ThreadingCtx` を使う。`multiprocessing.Queue.empty()` は同一 process 内で信頼できないため

### Running Tests

```bash
# Fast path（CI と同じ対象範囲）
uv run pytest -m "not slow and not veryslow"

# coverage 付き
uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report=term-missing

# integration test（MuJoCo が必要）
uv run pytest -m "slow and not veryslow" -v

# 完全な学習 smoke test
uv run pytest -m veryslow -v
```

## CI Workflow

`main` 向け PR では 3 つの job が自動実行されます。現在の workflow では、PR が `main` に merge された後に同じ CI 一式を再実行しません。

| Job | 内容 | 失敗で block するか |
|-----|------|---------------------|
| `lint` | `ruff check` + `ruff format --check` | ✅ |
| `typecheck` | `mypy src/unilab` + `pyright` | ✅ |
| `test` | `pytest -m "not slow and not veryslow" --cov --cov-fail-under=10` | ✅ |

`*.md`、`docs/**`、issue templates、`CODEOWNERS` のような docs-only / 協業メタデータ変更では CI は起動しません。

## Documentation Expectations

- ドキュメント内の各コマンドは、現在のリポジトリ内の実在する script、config、Makefile target と一致している必要があります
- backend support を記述する際は、`Registered`、`Configured`、`Benchmarked`、`Recommended` を優先して使ってください
- GitHub 上で正しくレンダリングされるよう、相対リンクを使ってください
- ユーザー向け docs を変更した場合は、English、zh_CN、Japanese、Korean の 4 系統を構造的に厳密に揃えてください
- CI、ログルート、support matrix に触れる場合は、`.github/workflows/ci.yml`、`scripts/`、`conf/` と照合してください

## GitHub Collaboration Model

- **Issue**: issue 1 件につき実行可能な work item 1 件
- **Milestone**: `M1` のようなフェーズ目標
- **PR**: driving issue を必ず link し、検証コマンドと影響範囲を記載
- **CODEOWNERS**: 実行 owner ではなく review owner を表す

詳しい協業ルールは [06-collaboration.md](06-collaboration.md) を参照してください。

## Pull Request Workflow

1. コードまたは config を変更した場合は、ローカルで `make check` を実行し、lint、mypy、pyright を通してください。
2. コード変更がある場合は、ローカルで `make test` を実行し、non-slow テストを通してください。
3. IPC、Runner、Config を触った場合は、対応するテストを追加または更新してください。
4. docs-only 変更では、少なくとも Markdown リンク、ファイルパス、script 名、コマンド引数を再確認してください。
5. 対応する GitHub issue を link し、PR template に検証内容と影響範囲を記入してください。
6. `main` 向けに PR を作成し、CI が green になるのを待ってください。
7. code review を待ってください。

## Issue Reports

bug 報告や機能提案には GitHub Issues を使ってください。

## Configuration System

UniLab は Hydra + dataclass 構成を使っています:

- **新しい task を追加する**: `conf/{algo}/task/` に YAML を作成し、`# @package _global_` を使う
- **hyperparameter を変更する**: 対応する YAML を編集するか、`algo.num_envs=2048` のような CLI override を使う
- **新しい algorithm を追加する**: `structured_configs.py` に dataclass を追加し、対応する `conf/` ディレクトリを作る

詳細は [Training Guide](03-training.md) の Hydra 節と [Development Architecture](00-development-architecture.md) を参照してください。
