# 代理环境下的安装指南

本文档记录在使用网络代理（HTTP / SOCKS5）的环境下安装 UniLab 时遇到的典型问题、根因分析与解决方案。适用于国内开发者或任何需要通过代理访问外网的场景。

## 背景知识

### uv 与 conda 的关系

UniLab 使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖。`uv sync` 会在项目根目录下创建独立的 `.venv/` 虚拟环境，**与 conda 环境互不干扰**。

```
conda env (unilab)          ← 提供 Python 解释器 + uv 可执行文件
  └── UniLab/.venv/         ← uv sync 创建的项目虚拟环境，所有依赖装在这里
```

- conda 环境只是外层隔离，提供干净的 Python 版本和 `uv` 命令。
- 所有项目依赖通过 `uv sync` 安装到 `.venv/`，训练命令通过 `uv run` 执行。
- **手动 `uv pip install` 时，如果当前激活了 conda 环境，包会默认装到 conda 而不是 `.venv/`**——这是一个常见陷阱。

### UniLab 的包源结构

UniLab 的依赖来自三个不同的包源：

| 包源 | 用途 | 示例包 |
|------|------|--------|
| PyPI (pypi.org) | 大部分通用依赖 | numpy, torch, gymnasium |
| PyTorch cu128 索引 | CUDA 版 PyTorch wheel | torch==2.7.0+cu128 |
| Motphys 私有 PyPI (`pypi.motphys.com`) | MotrixSim 物理引擎 | motrixsim-core |

这三个包源对代理的需求各不相同，这正是问题根源。

---

## 问题 1：motrixsim-core 下载超时

### 现象

```
× Failed to download `motrixsim-core==0.8.1.dev104665`
├─▶ Request failed after 3 retries in 49.5s
├─▶ Failed to fetch:
│   `https://pypi.motphys.com/packages/motrixsim_core-...whl`
├─▶ error sending request for url
╰─▶ operation timed out
```

### 根因

代理服务器（如 Clash、V2Ray 等）转发到 `pypi.motphys.com`（Motphys 私有 PyPI）时超时。可能的原因：

1. **代理路由不当**：代理将 `pypi.motphys.com` 的流量发往海外节点，但该服务器部署在国内，经过代理反而变慢或不可达。
2. **代理协议不兼容**：某些代理对非标准端口或私有证书的 HTTPS 连接处理异常。
3. **DNS 解析冲突**：代理的远端 DNS 解析与本地 DNS 解析到不同的服务器 IP。

### 解决方案

通过 `no_proxy` 环境变量让 `pypi.motphys.com` 绕过代理、直连访问：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy="pypi.motphys.com"
export NO_PROXY="pypi.motphys.com"

make setup-motrix
```

> **为什么要同时设 `no_proxy` 和 `NO_PROXY`？**
>
> 不同的工具和库读取不同的变量名。`curl` 和多数 Linux 工具读 `no_proxy`（小写），部分 Java / Go 工具读 `NO_PROXY`（大写）。`uv` 内部使用 Rust 的 HTTP 客户端，两者都会检查。同时设置可确保所有工具链都能正确绕过。

### 判断依据

如果你的环境**不使用代理**或 `pypi.motphys.com` 能直接访问，则不会遇到此问题。可以用以下命令测试：

```bash
# 不走代理直连测试
curl --noproxy '*' -I https://pypi.motphys.com/simple/
# 走代理测试
curl -x http://127.0.0.1:7897 -I https://pypi.motphys.com/simple/
```

如果直连成功但走代理失败，就需要 `no_proxy` 设置。

---

## 问题 2：httpx SOCKS 代理缺少 socksio

### 现象

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
Make sure to install httpx using `pip install httpx[socks]`
```

### 根因

UniLab 的依赖 `huggingface_hub` 使用 `httpx` 作为 HTTP 客户端，用于从 Hugging Face 下载模型和数据（如 demo checkpoint、grasp cache 等）。

当系统配置了 SOCKS5 代理时（通常通过 `all_proxy` 或 `ALL_PROXY` 环境变量），`httpx` 需要额外的 `socksio` 包来处理 SOCKS 协议。但 `httpx` 的默认安装不包含 `socksio`，UniLab 的 `pyproject.toml` 也没有将其列为依赖（因为并非所有用户都需要代理）。

**代理协议说明：**

| 协议 | 环境变量示例 | 常见软件 |
|------|-------------|---------|
| HTTP 代理 | `http_proxy=http://127.0.0.1:7897` | 多数场景，httpx 原生支持 |
| SOCKS5 代理 | `all_proxy=socks5://127.0.0.1:7897` | Clash/V2Ray 的 SOCKS 端口，需 socksio |

很多代理软件（如 Clash）同时监听 HTTP 和 SOCKS5 端口。如果你的 shell 配置（`~/.bashrc`、`~/.zshrc`）中设了 `all_proxy=socks5://...`，即使 `http_proxy` 用的是 HTTP 协议，`httpx` 也会尝试走 SOCKS5。

### 解决方案

**方案 A：安装 socksio（推荐）**

```bash
uv pip install httpx[socks] --python .venv/bin/python
```

> **关键：必须指定 `--python .venv/bin/python`**
>
> 如果当前激活了 conda 环境，`uv pip install` 默认会将包安装到 conda 环境的 site-packages，而非项目的 `.venv/`。加上 `--python .venv/bin/python` 确保安装到正确位置。
>
> 验证安装位置：
> ```bash
> uv run python -c "import socksio; print(socksio.__file__)"
> # 应该输出 .venv/ 下的路径
> ```

**方案 B：改用 HTTP 代理（避免 SOCKS）**

如果你的代理软件同时提供 HTTP 和 SOCKS5 端口，可以只使用 HTTP 协议，不需要 socksio：

```bash
# 只设 http/https 代理，不设 all_proxy
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
unset all_proxy
unset ALL_PROXY
```

检查当前环境变量：

```bash
env | grep -i proxy
```

如果输出中有 `all_proxy=socks5://...` 或 `ALL_PROXY=socks5://...`，这就是触发 SOCKS 报错的来源。

---

## 问题 3：SAC / TD3 训练立即退出（exit code 247）

### 现象

```
Motphys profiler initialized: disabled
resource_tracker: There appear to be 5 leaked semaphore objects to clean up at shutdown
```

只有这两行输出，然后进程退出，exit code 247。没有 Python traceback。

### 根因

SAC / TD3 等 off-policy 算法需要 JIT 编译一个 C++ CUDA 扩展 `unilab_native_h2d`（用于高效 host-to-device 数据搬运）。编译在 collector subprocess 中触发，如果编译失败，subprocess 直接 crash，父进程只收到退出信号，错误信息被吞掉。

用 `--sim mujoco` 运行相同任务可以看到完整 traceback（mujoco 变体不使用 subprocess collector），从而定位根因。

这个编译过程需要三个前置条件，缺任何一个都会失败：

### 3a. 缺少 C++ 编译器

**报错**（通过 mujoco 变体可见）：

```
/bin/sh: 1: c++: not found
```

**解决**：

```bash
sudo apt-get install build-essential
```

这会安装 `gcc`、`g++`、`make` 等编译工具链。

### 3b. 缺少 CUDA Toolkit 头文件

**报错**：

```
fatal error: cuda_runtime_api.h: 没有那个文件或目录
```

**原因**：系统只有 NVIDIA 驱动（`nvidia-smi` 能用），没有 CUDA Toolkit。PyTorch 的 pip wheel 虽然包含 CUDA runtime 库（`.so`），但不包含编译所需的头文件（`.h`）。JIT 编译需要完整 CUDA Toolkit。

**解决**：

```bash
# 先给 conda 配代理（见问题 4），然后安装
conda install -n unilab -c nvidia cuda-toolkit=12.8 -y
```

> **驱动 vs Toolkit 的区别**：
> - **NVIDIA 驱动**：让 GPU 能工作，`nvidia-smi` 能用。随系统或 `.run` 安装。
> - **CUDA Toolkit**：包含 `nvcc` 编译器、`cuda_runtime_api.h` 等头文件、cuBLAS/cuDNN 等开发库。通过 `conda`、`apt` 或 NVIDIA 官网安装。
> - PyTorch pip wheel 自带运行时 `.so`，跑 PPO/APPO 不需要 Toolkit；但 SAC/TD3 的 JIT 编译需要。

### 3c. conda CUDA Toolkit 头文件路径不标准

**报错**：安装了 CUDA Toolkit 但仍然 `cuda_runtime_api.h: 没有那个文件`。

**原因**：通过 conda 安装的 CUDA Toolkit 把头文件放在 `$CONDA_PREFIX/targets/x86_64-linux/include/` 而非 PyTorch JIT 期望的 `$CUDA_HOME/include/`。

**解决**：

```bash
export CUDA_HOME=$CONDA_PREFIX  # 即 /home/<user>/anaconda3/envs/unilab
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
```

设好后，JIT 编译结果会缓存到 `~/.cache/torch_extensions/`，后续启动不再重新编译。

---

## 问题 4：conda install 连接失败

### 现象

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
<https://conda.anaconda.org/nvidia/linux-64/gds-tools-1.13.1.3-0.conda>
```

### 根因

conda 有自己的网络栈，**不读取 shell 的 `http_proxy` / `https_proxy` 环境变量**。即使 shell 中配了代理，conda 仍然直连，在需要代理的网络中就会连接失败。

### 解决方案

**方案 A：给 conda 配代理（推荐）**

```bash
conda config --set proxy_servers.http http://127.0.0.1:7897
conda config --set proxy_servers.https http://127.0.0.1:7897
```

这会写入 `~/.condarc`，全局生效。

**方案 B：在 `.condarc` 中直接编辑**

```yaml
# ~/.condarc
proxy_servers:
  http: http://127.0.0.1:7897
  https: http://127.0.0.1:7897
```

**方案 C：使用清华 conda 镜像**

如果代理不稳定，可以改用国内镜像源避开代理：

```yaml
# ~/.condarc
channels:
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

> **注意**：nvidia channel 的清华镜像可能不包含最新版本的 `cuda-toolkit`。如果镜像版本不够新，仍需使用方案 A（配代理）从官方源安装。

---

## 完整安装流程（代理环境）

综合以上所有问题，国内代理环境下的推荐安装流程：

```bash
# 0. 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. 创建 conda 环境
conda create -n unilab python=3.13
conda activate unilab
pip install uv

# 2. 给 conda 配代理（根据你的端口修改）
conda config --set proxy_servers.http http://127.0.0.1:7897
conda config --set proxy_servers.https http://127.0.0.1:7897

# 3. 安装系统依赖
sudo apt-get install build-essential cmake ffmpeg

# 4. 安装 CUDA Toolkit（SAC/TD3 等 off-policy 算法需要）
conda install -c nvidia cuda-toolkit=12.8 -y

# 5. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 6. 配置代理环境变量（根据你的代理端口修改）
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy="pypi.motphys.com"
export NO_PROXY="pypi.motphys.com"
unset all_proxy
unset ALL_PROXY

# 7. 安装 Python 依赖
make setup-motrix

# 8. 如果需要 SOCKS 代理支持
uv pip install httpx[socks] --python .venv/bin/python

# 9. 配置运行时环境变量
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
export HF_ENDPOINT=https://hf-mirror.com

# 10. 验证安装
uv run python -c "
import torch
print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import unilab; print('unilab OK')
import mujoco; print(f'mujoco {mujoco.__version__}')
try:
    import motrixsim; print('motrixsim OK')
except ImportError:
    print('motrixsim not installed (optional)')
"

# 11. 激活 shell 补全
source ~/.bashrc

# 12. 冒烟测试
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 algo.num_envs=16 training.no_play=true

# 13. 运行 demo
uv run demo dance
```

## 持久化环境配置

将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`，避免每次手动 export：

```bash
# UniLab 代理配置
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy="pypi.motphys.com,localhost,127.0.0.1"
export NO_PROXY="pypi.motphys.com,localhost,127.0.0.1"

# CUDA Toolkit（conda 安装路径）
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

> **注意**：`$CONDA_PREFIX` 只有在 conda 环境激活后才有值。如果 shell 启动时未自动激活 conda 环境，需要写完整路径：
> ```bash
> export CUDA_HOME=/home/<user>/anaconda3/envs/unilab
> export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include
> ```

## 常见排查命令

```bash
# 查看当前所有代理相关环境变量
env | grep -i proxy

# 测试 PyPI 连通性（走代理）
curl -I https://pypi.org/simple/

# 测试 Motphys PyPI 连通性（直连）
curl --noproxy '*' -I https://pypi.motphys.com/simple/

# 测试 HuggingFace 连通性
curl -I https://huggingface.co

# 检查 conda 代理配置
conda config --show proxy_servers

# 检查 CUDA Toolkit
nvcc --version
ls $CUDA_HOME/targets/x86_64-linux/include/cuda_runtime_api.h

# 检查 C++ 编译器
g++ --version

# 检查 .venv 中已安装的包
uv pip list --python .venv/bin/python | grep -E "torch|mujoco|motrix|socksio|httpx"

# 检查 JIT 编译缓存
ls ~/.cache/torch_extensions/

# 检查 uv pip install 的默认目标环境
uv pip install --dry-run httpx[socks]  # 看 "Using Python ... environment at: ..."
```

## 问题速查表

| 现象 | 根因 | 解决 |
|------|------|------|
| `motrixsim-core` 下载超时 | 代理无法转发 motphys.com | `no_proxy=pypi.motphys.com` |
| `socksio` not installed | SOCKS5 代理 + httpx | `uv pip install httpx[socks] --python .venv/bin/python` |
| SAC/TD3 立即退出 (247) | subprocess JIT 编译失败 | 用 `--sim mujoco` 看完整报错 |
| `c++: not found` | 缺编译器 | `sudo apt-get install build-essential` |
| `cuda_runtime_api.h` 缺失 | 缺 CUDA Toolkit | `conda install -c nvidia cuda-toolkit=12.8` |
| CUDA headers 仍找不到 | conda 头文件路径不标准 | `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| conda HTTP 000 连接失败 | conda 不读 shell 代理 | `conda config --set proxy_servers.https http://127.0.0.1:7897` |
| PPO/APPO 正常但 SAC 不行 | on-policy 不需要 JIT | 只有 off-policy 需要 CUDA Toolkit + g++ |
| `ffmpeg` not found (视频导出) | 缺 ffmpeg 系统包 | `sudo apt install ffmpeg` |
