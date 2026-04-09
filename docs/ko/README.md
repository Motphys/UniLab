![G1 motion tracking overview](../assets/g1_readme.png)

# UniLab

언어: [English](../../README.md) | [简体中文](../zh_CN/README.md) | [日本語](../ja/README.md) | 한국어

UniLab의 핵심 목표는 하나의 가설을 검증하는 것입니다. **robot locomotion RL은 GPU 시뮬레이션 백엔드에 의존하지 않아도 된다**는 가설입니다.

주류 프레임워크는 물리 시뮬레이션, replay buffer, 정책 학습을 하나의 GPU pipeline에 강하게 결합하는 경우가 많습니다. UniLab은 다른 경로를 택합니다. **CPU 시뮬레이션 + shared-memory 데이터 경로 + GPU 학습**입니다. 이 방식은 학습 처리량을 유지하면서도 특정 시뮬레이터나 하드웨어 플랫폼에 대한 결합을 줄여 줍니다.

```text
┌───────────────────┐     통합 shared memory 경로    ┌────────────────────┐
│   CPU 물리 시뮬    │ ───────────────────────────▶  │   GPU 정책 학습    │
│  mujoco.rollout   │       SharedReplayBuffer      │   PPO / SAC / TD3  │
│   멀티스레드 step  │    (PyTorch shared tensors)   │     CUDA / MPS     │
└───────────────────┘                                └────────────────────┘
```

- **CPU simulation**: MuJoCo / Motrix의 CPU 멀티스레드 step을 사용하며 GPU sim kernel이 필요 없습니다
- **통합 메모리 경로**: collector와 learner가 PyTorch shared tensors로 zero-copy 통신합니다
- **GPU training**: 정책 네트워크 학습은 여전히 GPU에서 수행되며 CUDA와 MPS를 지원합니다
- **하드웨어 비종속**: macOS와 Linux 모두를 1급 개발 환경으로 다룹니다

## Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 의존성 설치
# macOS (MPS, PyPI torch wheel 기본 설치)
uv sync

# Linux (기본값: PyTorch cu128 wheel 설치)
# 현재 PyTorch cu128 wheel 이 지원하는 NVIDIA GPU 및 드라이버 스택이 필요함
uv sync

# 선택 사항: Motrix 백엔드
uv sync --extra motrix

# 3. 학습 실행
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

## Workflow Entrypoints

| 목표 | 엔트리포인트 | 기본 로그 루트 |
|------|--------------|----------------|
| PPO (torch / RSL-RL) | `scripts/train_rsl_rl.py` | `logs/rsl_rl_train/<task>/` |
| PPO (MLX, macOS) | `scripts/train_mlx_ppo.py` | `logs/mlx_rl_train/<task>/` |
| APPO | `scripts/train_appo.py` | `logs/appo/<task>/` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `logs/fast_sac/<task>/` / `logs/fast_td3/<task>/` |

학습 스크립트는 기본적으로 학습이 끝난 뒤 자동 playback으로 들어갑니다. `training.no_play=true`를 지정하면 건너뛸 수 있습니다.

## Repository Map

- `conf/`: Hydra 설정과 task / reward / algorithm 조합
- `scripts/`: 학습, playback, motion 전처리, 각종 도구의 직접 엔트리포인트
- `src/unilab/`: 환경, 백엔드, 알고리즘, 공용 유틸리티
- `tests/`: unit test, integration test, 스크립트 설정 테스트
- `docs/`: `docs/en/`, `docs/zh_CN/`, `docs/ja/`, `docs/ko/`로 나뉜 다국어 문서

## Documentation

- [00 RL Infrastructure Development Standard](00-development-architecture.md): 설계 원칙, 계층 구조, contract, 검증 경계
- [01 Getting Started](01-getting-started.md): 설치, 의존성 동기화, 미러, 첫 실행 명령
- [02 Simulation Backends](02-simulation-backends.md): MuJoCo / Motrix 지원 범위와 백엔드 선택
- [03 Training Guide](03-training.md): 학습, playback, 재개, Hydra override, W&B
- [04 Algorithms](04-algorithms.md): APPO, FastSAC, FastTD3의 사용법과 차이
- [05 G1 Motion Tracking](05-g1-motion-tracking.md): G1 전신 motion tracking 작업
- [06 Collaboration Workflow](06-collaboration.md): GitHub issue / milestone / PR 협업 규칙
- [Contributing](CONTRIBUTING.md): 개발 흐름, 테스트, CI, review 기대치
- [AGENTS](../../AGENTS.md): 이 RL infra 저장소에서 작업하는 coding agent / 자동 편집기를 위한 가이드

## Related Projects

1. https://github.com/mujocolab/mjlab
2. https://github.com/amazon-far/holosoma
3. https://github.com/google-deepmind/mujoco
4. https://github.com/google-deepmind/mujoco_playground/
