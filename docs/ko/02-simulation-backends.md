# 시뮬레이션 백엔드

언어: [English](../en/02-simulation-backends.md) | [简体中文](../zh_CN/02-simulation-backends.md) | [日本語](../ja/02-simulation-backends.md) | 한국어

UniLab은 현재 두 개의 simulation backend를 지원합니다:

- **MuJoCo**: 기본 백엔드이며 기능 범위가 가장 넓습니다
- **Motrix**: 선택적 백엔드이며 task와 algorithm 지원을 계속 보강하는 중입니다

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

범례:

- `✅` 지원됨
- `⚠️` 진행 중

## Select A Backend

기본 backend는 `mujoco`입니다. Hydra parameter `training.sim_backend`로 `motrix`로 전환합니다.

```bash
# 기본 MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick

# 명시적으로 Motrix 지정
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix
```

## Playback Differences

- `mujoco`: 학습 후 자동 playback에서 `play_video.mp4`를 내보냅니다
- `motrix`: playback은 보통 비디오를 저장하지 않고 대화형 renderer 창을 엽니다

G1 motion tracking에서 현재 검증된 Motrix 경로는 `PPO (torch) + motrix`와 `APPO (torch) + motrix`입니다. `scripts/play_interactive.py`는 아직 MuJoCo 경로를 따릅니다.

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## Notes

- backend 지원 범위는 단계별 capability snapshot이므로 일시적인 실행 상태를 top-level README의 주장으로 올리지 마세요
- 진행 상황은 저장소 내부의 임시 목록이 아니라 GitHub milestone과 issue로 추적하세요

## Navigation

- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)
