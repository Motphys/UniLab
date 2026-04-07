# G1 Motion Tracking

언어: [English](../en/05-g1-motion-tracking.md) | [简体中文](../zh_CN/05-g1-motion-tracking.md) | [日本語](../ja/05-g1-motion-tracking.md) | 한국어

UniLab은 현재 G1용 전신 motion tracking task를 하나 제공합니다.

- Hydra task: `g1_motion_tracking`
- 등록된 env 이름: `G1MotionTracking`
- 등록된 backend: `mujoco`와 `motrix`
- 반영된 Motrix 전용 config: PPO와 APPO의 motion-tracking reward
- 기본 motion 파일: `src/unilab/assets/motions/g1/dance1_subject2_part.npz`

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

# 최신 checkpoint 재생
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.play_only=true

# Motrix PPO playback은 native renderer를 엽니다
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true

# APPO MuJoCo playback
uv run python scripts/train_appo.py task=g1_motion_tracking training.play_only=true

# APPO Motrix playback은 native renderer를 엽니다
uv run python scripts/train_appo.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true
```

G1 motion tracking에서 Motrix 학습과 playback의 주 경로는 `scripts/train_rsl_rl.py`와 `scripts/train_appo.py`입니다. 디버그용 스크립트 `scripts/play_interactive.py`는 아직 MuJoCo viewer 경로를 따릅니다.

## Interactive Debugging

`scripts/play_interactive.py`는 target body를 직접 시각화할 수 있고 reward가 사용하는 기준 pose와 velocity도 보여줄 수 있습니다. 이 스크립트는 MuJoCo viewer 기준으로 구현되어 있으며 Motrix native renderer는 지원하지 않습니다.

```bash
# motion target 시각화
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_show_axes

# 일부 body만 보기
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_body_names torso_link,left_wrist_yaw_link,right_wrist_yaw_link

# reward debug 정보 보기
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_reward_debug \
  --reward_debug_show_velocity \
  --reward_debug_show_connectors \
  --target_max_bodies 4
```

특정 run 또는 checkpoint가 필요하면 `--load_run`과 `--checkpoint`도 함께 전달하세요.

## Motion Preprocessing

학습 환경은 전처리된 `.npz` 파일을 읽습니다. `scripts/motion/csv_to_npz.py`를 사용하면 Unitree 형식 CSV를 학습 가능한 NPZ로 변환할 수 있습니다:

```bash
# 전체 변환
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50

# 일부 시간 구간만 내보내기
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

## Replay NPZ

NPZ를 만든 뒤에는 `scripts/motion/replay_npz.py`로 MuJoCo viewer에서 바로 확인할 수 있습니다:

```bash
# 루프 재생
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop

# 0.5x 슬로우 재생
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

## Configuration Note

`task=g1_motion_tracking`는 기본적으로 env config에 선언된 `motion_file`을 읽습니다. custom motion으로 바꾸려면 먼저 `.npz`를 생성하고 env config의 기본 `motion_file`을 갱신하세요.

Motrix 경로를 검증할 때는 MuJoCo 전용 디버그 스크립트보다 학습 스크립트에 내장된 play mode를 우선 사용하세요:

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)
