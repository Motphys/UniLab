# 알고리즘

언어: [English](../en/04-algorithms.md) | [简体中文](../zh_CN/04-algorithms.md) | [日本語](../ja/04-algorithms.md) | 한국어

이 페이지는 algorithm 수준 설명만 다룹니다. 엔트리포인트와 공통 CLI 파라미터는 [03-training.md](03-training.md)를 참고하세요.

## APPO

APPO는 V-trace importance-sampling 보정을 포함한 UniLab의 비동기 PPO 구현입니다. collector subprocess가 CPU simulation을 담당하고 learner process가 GPU training을 담당하며 ring-buffer 파이프라인으로 병렬 실행됩니다.

### Core Features

| 특징 | 설명 |
|------|------|
| 비동기 멀티프로세스 | collector와 learner가 병렬로 동작 |
| V-trace IS 보정 | `pi_target / pi_behavior`로 off-policy 데이터를 보정 |
| 4-slot ring buffer | 최대 4개의 rollout이 동시에 진행 가능 |
| Replay queue | learner 측에서 대기 중 rollout을 보관하는 queue |
| 로그 디렉터리 | `logs/appo/<task>/<timestamp>_mujoco/` |

### Usage

```bash
# 기본 학습
uv run python scripts/train_appo.py task=go1_joystick

# 환경 수와 iteration 수 지정
uv run python scripts/train_appo.py task=go2_joystick algo.num_envs=2048 algo.max_iterations=300

# replay queue 깊이 조정
uv run python scripts/train_appo.py task=go1_joystick training.replay_queue_size=2

# 자동 playback 건너뛰기
uv run python scripts/train_appo.py task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true
uv run python scripts/train_appo.py task=go1_joystick training.play_only=true training.load_run="2026-03-16_01-35-12_mujoco"
```

### Key Parameters

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `task` | `go2_joystick` | task config 이름 |
| `algo.max_iterations` | 150 | 최대 학습 iteration 수 |
| `algo.num_envs` | 2048 | 병렬 env 수 |
| `algo.steps_per_env` | 24 | env당 rollout 길이 |
| `training.replay_queue_size` | 3 | learner 측 rollout replay 깊이 |
| `training.device` | 자동 감지 | learner device |
| `training.collector_device` | `cpu` | collector device |
| `training.logger` | `tensorboard` | logging backend |
| `training.play_only` | false | playback만 수행 |
| `training.no_play` | false | 자동 playback 건너뜀 |
| `training.load_run` | `-1` | run 디렉터리 이름 또는 checkpoint path |
| `algo.save_interval` | 50 | checkpoint 저장 간격 |

### APPO vs PPO

| 관점 | rsl-rl PPO | APPO |
|------|------------|------|
| 수집 방식 | 동기 | 비동기 |
| IS 보정 | 없음 | V-trace |
| CPU / GPU 활용률 | 번갈아 포화 | 동시에 포화 |
| 적합한 경우 | sample efficiency 우선 | throughput 우선 |

## FastSAC And FastTD3

FastSAC과 FastTD3는 동일한 비동기 멀티프로세스 구조를 사용해 shared memory로 CPU simulation과 GPU training을 분리합니다.

### Core Features

| 특징 | 설명 |
|------|------|
| 비동기 멀티프로세스 | collector와 learner가 독립적으로 동작 |
| 통합 shared memory | PyTorch shared tensors를 이용한 zero-copy 전송 |
| 동기 / 비동기 모드 | 기본 동기 수집과 비동기 수집을 모두 지원 |
| 자동 playback | 학습 후 자동으로 playback 실행 |

### Usage

```bash
# 기본 학습
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick

# 비동기 수집 모드
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.no_sync_collection=true

# 자동 playback 건너뛰기
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.no_play=true
```

### Playback

```bash
uv run python scripts/train_offpolicy.py algo=sac task=go2_joystick training.play_only=true
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick training.play_only=true training.load_run="2024-02-04_12-00-00"
```

### Key Parameters

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `algo` | `sac` | algorithm 선택 |
| `task` | `go1_joystick` | task config 이름 |
| `algo.max_iterations` | 500 (SAC) / 5000 (TD3) | 최대 학습 iteration 수 |
| `algo.num_envs` | 4096 | 병렬 env 수 |
| `training.device` | 자동 감지 | learner device |
| `training.sim_backend` | `mujoco` | simulation backend |
| `training.no_sync_collection` | false | 비동기 수집 활성화 |
| `training.env_steps_per_sync` | 1 | 동기 모드에서 한 번에 수집하는 step 수 |
| `training.play_only` | false | playback만 수행 |
| `training.no_play` | false | 자동 playback 건너뜀 |

## Navigation

- Previous: [Training Guide](03-training.md)
- Next: [G1 Motion Tracking](05-g1-motion-tracking.md)
