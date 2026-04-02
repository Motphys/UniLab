# Getting Started

本页只回答三件事：

1. 怎么把 UniLab 跑起来
2. macOS 和 Linux 各自怎么装
3. 第一次该跑什么命令确认环境正常

## Install

### 使用 uv

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 3. 安装系统依赖
brew install cmake  # macOS
# sudo apt-get install cmake  # Ubuntu / Debian
```

### 同步依赖

```bash
# macOS (MPS)
uv sync --extra dev

# Linux (CUDA 12.4)
uv sync --extra dev --extra cu124

# 可选：Motrix 后端
uv sync --extra motrix
```

## 国内镜像

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## First Run

### 训练一个最小任务

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick
```

### 只做回放

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

### 运行检查

```bash
make check
uv run pytest -m "not slow"
```

## Next

- 训练和回放细节见 [training.md](training.md)
- 后端差异见 [simulation-backends.md](simulation-backends.md)
- 算法说明见 [algorithms.md](algorithms.md)
