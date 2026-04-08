# 시작하기

언어: [English](../en/01-getting-started.md) | [简体中文](../zh_CN/01-getting-started.md) | [日本語](../ja/01-getting-started.md) | 한국어

이 페이지는 세 가지 질문에만 답합니다:

1. UniLab을 어떻게 실행하는가?
2. macOS와 Linux의 설치 단계는 어떻게 다른가?
3. 환경이 정상인지 확인하려면 처음에 어떤 명령을 실행해야 하는가?

## Install

### uv 사용

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 저장소 클론
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 3. 시스템 의존성 설치
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu / Debian
```

### 의존성 동기화

```bash
# macOS (MPS)
uv sync

# Linux (CUDA 11.8 / 12.4 / 12.6 / 12.8)
uv sync --extra cu118
uv sync --extra cu124
uv sync --extra cu126
uv sync --extra cu128

# 선택 사항: Motrix 백엔드
uv sync --extra motrix
uv sync --extra cu124 --extra motrix
```

## 중국 본토 미러

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### 최소 task 학습

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

### 자주 쓰는 엔트리포인트

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py task=go1_joystick

# APPO
uv run python scripts/train_appo.py task=go1_joystick

# SAC / TD3
uv run python scripts/train_offpolicy.py algo=sac task=go1_joystick
uv run python scripts/train_offpolicy.py algo=td3 task=go1_joystick
```

### 환경 검증

```bash
make check
uv run pytest -m "not slow and not veryslow"
```

## Navigation

- Previous: [Development Architecture](00-development-architecture.md)
- Next: [Simulation Backends](02-simulation-backends.md)
