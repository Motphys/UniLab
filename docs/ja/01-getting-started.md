# はじめに

言語: [English](../en/01-getting-started.md) | [简体中文](../zh_CN/01-getting-started.md) | 日本語 | [한국어](../ko/01-getting-started.md)

このページは次の 3 点だけに答えます:

1. UniLab をどう起動するか
2. macOS と Linux でインストール手順がどう違うか
3. 環境確認のために最初に何を実行すべきか

## Install

### uv を使う

```bash
# 1. uv をインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. リポジトリを clone
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 3. システム依存を入れる
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu / Debian
```

### 依存関係を同期

```bash
# macOS (MPS)
uv sync

# Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8)
uv sync --extra cu118
uv sync --extra cu124
uv sync --extra cu126
uv sync --extra cu128

# オプション: Motrix backend
uv sync --extra motrix
uv sync --extra cu124 --extra motrix
```

## 中国本土向けミラー

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### 最小タスクを学習する

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

### よく使うエントリポイント

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# SAC / TD3
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick
```

### 環境を検証する

```bash
make check
uv run pytest -m "not slow and not veryslow"
```

## Navigation

- Previous: [Development Architecture](00-development-architecture.md)
- Next: [Simulation Backends](02-simulation-backends.md)
