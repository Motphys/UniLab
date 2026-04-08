![G1 motion tracking overview](../assets/g1_readme.png)

# UniLab

言語: [English](../../README.md) | [简体中文](../zh_CN/README.md) | 日本語 | [한국어](../ko/README.md)

UniLab の中核目標は、**robot locomotion RL は GPU シミュレーションバックエンドに依存しなくても成立する**、という仮説を検証することです。

一般的なフレームワークでは、物理シミュレーション、replay buffer、方策学習が 1 本の GPU pipeline に強く結合されています。UniLab は別の経路を取ります: **CPU シミュレーション + shared-memory データ経路 + GPU 学習**。これにより学習スループットを維持しながら、特定のシミュレータやハードウェアへの依存を下げます。

```text
┌───────────────────┐     統一 shared memory 経路    ┌────────────────────┐
│   CPU 物理シム    │ ────────────────────────────▶  │   GPU 方策学習     │
│  mujoco.rollout   │       SharedReplayBuffer       │   PPO / SAC / TD3  │
│   マルチスレッド  │    (PyTorch shared tensors)    │     CUDA / MPS     │
└───────────────────┘                                └────────────────────┘
```

- **CPU simulation**: MuJoCo / Motrix の CPU マルチスレッド step。GPU sim kernel は不要です
- **統一メモリ経路**: collector と learner が PyTorch shared tensors でゼロコピー通信します
- **GPU training**: 方策ネットワークの学習自体は GPU 上で行い、CUDA と MPS をサポートします
- **ハードウェア非依存**: macOS と Linux をどちらも第一級の開発環境として扱います

## Quick Start

```bash
# 1. リポジトリを clone
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 依存関係をインストール
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA extra を 1 つ選択。例: cu124)
uv sync --extra dev --extra cu124

# オプション: Motrix バックエンド
uv sync --extra dev --extra motrix

# 3. 学習ジョブを実行
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

## Workflow Entrypoints

| 目的 | エントリポイント | デフォルトのログルート |
|------|------------------|-------------------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

学習スクリプトは既定で学習終了後に自動再生へ入ります。`training.no_play=true` を指定するとスキップできます。

## Repository Map

- `conf/`: Hydra 設定と task / reward / algorithm の組み合わせ
- `scripts/`: 学習、再生、motion 前処理、各種ツールの直接エントリポイント
- `src/unilab/`: 環境、バックエンド、アルゴリズム、共通ユーティリティ
- `tests/`: unit test、integration test、スクリプト設定テスト
- `docs/`: `docs/en/`、`docs/zh_CN/`、`docs/ja/`、`docs/ko/` に分かれた多言語ドキュメント

## Documentation

- [00 RL Infrastructure Development Standard](00-development-architecture.md): 設計原則、レイヤ構成、contract、検証境界
- [01 Getting Started](01-getting-started.md): インストール、依存同期、ミラー、初回実行コマンド
- [02 Simulation Backends](02-simulation-backends.md): MuJoCo / Motrix の対応範囲とバックエンド選択
- [03 Training Guide](03-training.md): 学習、再生、再開、Hydra override、W&B
- [04 Algorithms](04-algorithms.md): APPO、FastSAC、FastTD3 の使い方と違い
- [05 G1 Motion Tracking](05-g1-motion-tracking.md): G1 全身 motion tracking タスク
- [06 Collaboration Workflow](06-collaboration.md): GitHub issue / milestone / PR の協業ルール
- [Contributing](CONTRIBUTING.md): 開発フロー、テスト、CI、review の期待値
- [AGENTS](../../AGENTS.md): この RL infra リポジトリで作業する coding agent / 自動編集器向けガイド

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
