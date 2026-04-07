# 学習ガイド

言語: [English](../en/03-training.md) | [简体中文](../zh_CN/03-training.md) | 日本語 | [한국어](../ko/03-training.md)

このページでは学習、playback、再開、Hydra override、W&B を扱います。

## Pick An Entrypoint

| 目的 | エントリポイント | デフォルトのログルート |
|------|------------------|-------------------------|
| PPO (RSL-RL / torch) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX / macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

## Start Training

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# PPO (MLX, Apple Silicon)
uv run python scripts/train_mlx_ppo.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# Off-policy
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# CLI override
uv run python scripts/train_offpolicy.py algo=sac task=g1_sac algo.num_envs=2048 algo.max_iterations=1000
```

学習スクリプトは既定で学習終了後に自動 playback へ入ります。

- `mujoco` は `play_video.mp4` を出力する
- `motrix` は対話型ウィンドウで描画する
- `training.no_play=true` で自動 playback をスキップできる

run ディレクトリ名は `YYYY-MM-DD_HH-MM-SS_<sim_backend>` 形式です。例: `2026-03-09_18-30-00_mujoco`。

## Playback

```bash
# 最新結果を再生
uv run python scripts/train_rsl_rl.py task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# 特定 run を再生
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

## Resume Training

```bash
uv run python scripts/train_rsl_rl.py task=go2_joystick training.load_run="2024-02-04_12-00-00"
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

すべての学習スクリプトは Hydra config で動いています。

```bash
# 一般形
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# よく使うパラメータ
task=go1_joystick
algo=sac
training.play_only=true
training.no_play=true
training.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

完全に合成された config は次で確認できます:

```bash
uv run python scripts/train_offpolicy.py --cfg job
```

## W&B

`training.logger=wandb` を設定すると Weights & Biases へ自動記録されます。学習スクリプトはローカルの run ディレクトリにも次のファイルを書き出します:

- `run_config.json`
- `run_summary.json`

backend が `mujoco` で、学習後に `play_video.mp4` が生成される場合は、その動画も現在の W&B run にアップロードされます。

```bash
# 基本利用
uv run python scripts/train_rsl_rl.py task=go1_joystick training.logger=wandb

# project / entity を共有
uv run python scripts/train_appo.py \
  task=go1_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# task ごとにグループ化
uv run python scripts/train_offpolicy.py \
  algo=sac \
  task=go2_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=go2_joystick
```

よく使うフィールド:

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

自動記録されるメタデータには task、algorithm、backend、device、hardware 情報、git 情報、完全な config、総実行時間、summary metrics、利用可能な場合は最終 playback video が含まれます。

## Navigation

- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
