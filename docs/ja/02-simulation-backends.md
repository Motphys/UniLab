# シミュレーションバックエンド

言語: [English](../en/02-simulation-backends.md) | [简体中文](../zh_CN/02-simulation-backends.md) | 日本語 | [한국어](../ko/02-simulation-backends.md)

UniLab は現在 2 つの simulation backend をサポートしています:

- **MuJoCo**: デフォルト backend。機能カバレッジが最も広い
- **Motrix**: オプション backend。task と algorithm の対応はまだ拡張中

## Support Matrix

### MuJoCo

| Algorithm | Go1 | Go2 | G1 |
|-----------|-----|-----|----|
| PPO (torch) | ✅ |  | ✅ |
| PPO (mlx) | ✅ |  | ✅ |
| SAC (torch) | ✅ | ⚠️ | ✅ |
| TD3 (torch) | ⚠️ | ⚠️ | ⚠️ |
| APPO (torch) | ✅ | ✅ | ✅ |

### Motrix

| Algorithm | Go1 | Go2 | G1 |
|-----------|-----|-----|----|
| PPO (torch) | ⚠️ |  | ✅ |
| PPO (mlx) | ⚠️ |  |  |
| SAC (torch) | ⚠️ |  |  |
| TD3 (torch) |  |  |  |
| APPO (torch) |  |  | ✅ |

凡例:

- `✅` 対応済み
- `⚠️` 開発中

## Select A Backend

デフォルト backend は `mujoco` です。Hydra parameter `training.sim_backend` で `motrix` に切り替えます。

```bash
# デフォルト MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick

# 明示的に Motrix
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix
```

## Playback Differences

- `mujoco`: 学習後の自動 playback で `play_video.mp4` を書き出す
- `motrix`: playback は通常、動画出力ではなく対話型 renderer ウィンドウを開く

G1 motion tracking で現在検証済みの Motrix 経路は `PPO (torch) + motrix` と `APPO (torch) + motrix` です。`scripts/play_interactive.py` はまだ MuJoCo 経路です。

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## Notes

- backend support はフェーズ時点の能力スナップショットなので、一時的な実行状況を top-level README の主張にしない
- 進捗はリポジトリ内の暫定一覧ではなく GitHub milestone / issue で追跡する

## Navigation

- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)
