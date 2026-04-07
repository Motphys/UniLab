# 학습 가이드

언어: [English](../en/03-training.md) | [简体中文](../zh_CN/03-training.md) | [日本語](../ja/03-training.md) | 한국어

이 페이지는 학습, playback, 재개, Hydra override, W&B를 다룹니다.

## Pick An Entrypoint

| 목표 | 엔트리포인트 | 기본 로그 루트 |
|------|--------------|----------------|
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

학습 스크립트는 기본적으로 학습이 끝난 뒤 자동 playback으로 들어갑니다.

- `mujoco`는 `play_video.mp4`를 내보냅니다
- `motrix`는 대화형 창으로 렌더링합니다
- `training.no_play=true`는 자동 playback을 건너뜁니다

run 디렉터리는 `YYYY-MM-DD_HH-MM-SS_<sim_backend>` 형식을 사용합니다. 예: `2026-03-09_18-30-00_mujoco`.

## Playback

```bash
# 최신 결과 재생
uv run python scripts/train_rsl_rl.py task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true

# 특정 run 재생
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

## Resume Training

```bash
uv run python scripts/train_rsl_rl.py task=go2_joystick training.load_run="2024-02-04_12-00-00"
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.load_run="2024-02-04_12-00-00"
```

## Hydra Overrides

모든 학습 스크립트는 Hydra config로 구동됩니다.

```bash
# 일반 형식
uv run python scripts/train_*.py [config_group=value] [key.subkey=value]

# 자주 쓰는 파라미터
task=go1_joystick
algo=sac
training.play_only=true
training.no_play=true
training.load_run="-1"
training.logger=tensorboard
algo.num_envs=2048
algo.max_iterations=1000
```

완전히 합성된 config는 다음으로 확인할 수 있습니다:

```bash
uv run python scripts/train_offpolicy.py --cfg job
```

## W&B

`training.logger=wandb`를 설정하면 Weights & Biases로 자동 기록됩니다. 학습 스크립트는 로컬 run 디렉터리에도 다음 파일을 씁니다:

- `run_config.json`
- `run_summary.json`

backend가 `mujoco`이고 학습 후 `play_video.mp4`가 생성되면 그 비디오도 현재 W&B run에 업로드됩니다.

```bash
# 기본 사용법
uv run python scripts/train_rsl_rl.py task=go1_joystick training.logger=wandb

# project / entity 공유
uv run python scripts/train_appo.py \
  task=go1_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_entity=my-team

# task별 그룹화
uv run python scripts/train_offpolicy.py \
  algo=sac \
  task=go2_joystick \
  training.logger=wandb \
  training.wandb_project=unilab-benchmark \
  training.wandb_group=go2_joystick
```

자주 쓰는 필드:

- `training.wandb_project`
- `training.wandb_entity`
- `training.wandb_group`
- `training.wandb_name`
- `training.wandb_tags`
- `training.wandb_notes`
- `training.wandb_mode=offline`

자동으로 기록되는 메타데이터에는 task, algorithm, backend, device, hardware 정보, git 정보, 전체 config, 총 실행 시간, summary metrics, 그리고 가능할 경우 최종 playback video가 포함됩니다.

## Navigation

- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Algorithms](04-algorithms.md)
