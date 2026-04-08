# アルゴリズム

言語: [English](../en/04-algorithms.md) | [简体中文](../zh_CN/04-algorithms.md) | 日本語 | [한국어](../ko/04-algorithms.md)

このページでは algorithm レベルの説明だけを扱います。エントリポイントや共通 CLI parameter は [03-training.md](03-training.md) を参照してください。

## APPO

APPO は V-trace importance-sampling 補正を持つ UniLab の非同期 PPO 実装です。collector subprocess が CPU simulation を、learner process が GPU training を担当し、ring buffer パイプラインで並列に動作します。

### Core Features

| 特徴 | 説明 |
|------|------|
| 非同期マルチプロセス | collector と learner が並列動作 |
| V-trace IS 補正 | `pi_target / pi_behavior` で off-policy データを補正 |
| 4 スロット ring buffer | 最大 4 本の rollout が同時に飛行可能 |
| Replay queue | learner 側で未消費 rollout を保持する queue |
| ログディレクトリ | `logs/appo/<task>/<timestamp>_mujoco/` |

### Usage

```bash
# デフォルト学習
uv run python scripts/train_appo.py task=go1_joystick

# 環境数と反復数を指定
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# replay queue 深さを調整
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# 自動 playback をスキップ
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| `task` | `go2_joystick` | task config 名 |
| `algo.max_iterations` | 150 | 最大学習反復数 |
| `algo.num_envs` | 2048 | 並列 env 数 |
| `algo.steps_per_env` | 24 | env ごとの rollout 長 |
| `training.replay_queue_size` | 3 | learner 側 rollout replay 深さ |
| `training.device` | 自動検出 | learner device |
| `training.collector_device` | `cpu` | collector device |
| `training.logger` | `tensorboard` | logging backend |
| `training.play_only` | false | playback のみ |
| `training.no_play` | false | 自動 playback をスキップ |
| `training.load_run` | `-1` | run ディレクトリ名または checkpoint path |
| `algo.save_interval` | 50 | checkpoint 保存間隔 |

### APPO vs PPO

| 観点 | rsl-rl PPO | APPO |
|------|------------|------|
| 収集方式 | 同期 | 非同期 |
| IS 補正 | なし | V-trace |
| CPU / GPU 利用率 | 交互に飽和 | 同時に飽和 |
| 向いている場面 | sample efficiency 優先 | throughput 優先 |

## FastSAC And FastTD3

FastSAC と FastTD3 は同じ非同期マルチプロセス構成を使い、shared memory で CPU simulation と GPU training を分離します。

### Core Features

| 特徴 | 説明 |
|------|------|
| 非同期マルチプロセス | collector と learner が独立して動く |
| 統一 shared memory | PyTorch shared tensors によるゼロコピー転送 |
| 同期 / 非同期モード | 既定の同期収集と非同期収集の両方をサポート |
| 自動 playback | 学習後に自動で playback を行う |

### Usage

```bash
# 基本学習
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# 非同期収集モード
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# 自動 playback をスキップ
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| `algo` | `sac` | algorithm 選択 |
| `task` | `go1_joystick` | task config 名 |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 最大学習反復数 |
| `algo.num_envs` | 4096 | 並列 env 数 |
| `training.device` | 自動検出 | learner device |
| `training.sim_backend` | `mujoco` | simulation backend |
| `training.no_sync_collection` | false | 非同期収集を有効化 |
| `training.env_steps_per_sync` | 1 | 同期モードで 1 回に収集する step 数 |
| `training.play_only` | false | playback のみ |
| `training.no_play` | false | 自動 playback をスキップ |

## Navigation

- Previous: [Training Guide](03-training.md)
- Next: [G1 Motion Tracking](05-g1-motion-tracking.md)
