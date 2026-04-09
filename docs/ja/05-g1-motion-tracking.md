# G1 Motion Tracking

言語: [English](../en/05-g1-motion-tracking.md) | [简体中文](../zh_CN/05-g1-motion-tracking.md) | 日本語 | [한국어](../ko/05-g1-motion-tracking.md)

UniLab は現在、G1 向けの全身 motion tracking task を 1 つ提供しています。

- Hydra task: `g1_motion_tracking`
- 登録済み env 名: `G1MotionTracking`
- 登録済み backend: `mujoco` と `motrix`
- 反映済み Motrix 専用 config: PPO と APPO の motion-tracking reward
- デフォルト motion file: `src/unilab/assets/motions/g1/dance1_subject2_part.npz`

## Environment Entrypoints

```bash
# PPO (RSL-RL, MuJoCo)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking

# PPO (RSL-RL, Motrix)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix

# APPO (MuJoCo)
uv run python scripts/train_appo.py task=g1_motion_tracking

# APPO (Motrix)
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix

# 最新 checkpoint を再生
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.play_only=true

# Motrix PPO の playback は native renderer を開く
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true

# APPO MuJoCo playback
uv run python scripts/train_appo.py task=g1_motion_tracking training.play_only=true

# APPO Motrix playback は native renderer を開く
uv run python scripts/train_appo.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true
```

G1 motion tracking で Motrix の学習と playback を行う場合は、主に `scripts/train_rsl_rl.py` と `scripts/train_appo.py` を使ってください。デバッグ用 script `scripts/play_interactive.py` はまだ MuJoCo viewer 前提です。

## Interactive Debugging

`scripts/play_interactive.py` では target body を直接可視化でき、reward が使う参照 pose と velocity も表示できます。この script は MuJoCo viewer 向け実装であり、Motrix native renderer はサポートしません。

```bash
# motion target を可視化
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_show_axes

# 一部の body のみを見る
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_body_names torso_link,left_wrist_yaw_link,right_wrist_yaw_link

# reward debug 情報を見る
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_reward_debug \
  --reward_debug_show_velocity \
  --reward_debug_show_connectors \
  --target_max_bodies 4
```

特定の run や checkpoint が必要な場合は `--load_run` と `--checkpoint` も追加してください。

## Motion Preprocessing

学習環境は前処理済み `.npz` を読みます。`scripts/motion/csv_to_npz.py` を使うと Unitree 形式の CSV を学習可能な NPZ に変換できます:

```bash
# 全量変換
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50

# 一部時間範囲だけ出力
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

## Replay NPZ

NPZ を作成したら、`scripts/motion/replay_npz.py` で MuJoCo viewer 上から直接確認できます:

```bash
# ループ再生
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop

# 0.5x スロー再生
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

## Configuration Note

`task=g1_motion_tracking` は既定で env config に定義された `motion_file` を読みます。custom motion に切り替える場合は、先に `.npz` を生成し、その後 env config の既定 `motion_file` を更新してください。

Motrix 経路を検証する場合は、MuJoCo 専用の debug script よりも、学習 script に組み込まれた play mode を優先してください:

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)
