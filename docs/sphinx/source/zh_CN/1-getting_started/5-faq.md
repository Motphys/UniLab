# 常见问题

安装与首次运行 UniLab 时的常见问题，按 现象 → 解决 组织。多数与代理环境
（HTTP / SOCKS5）、off-policy 算法的 JIT 编译依赖或视频导出相关。

## motrixsim-core 下载超时

```
× Failed to download `motrixsim-core==0.8.1.dev104665`
╰─▶ operation timed out
```

`pypi.motphys.com` 部署在国内，通过代理访问反而不可达。设 `no_proxy` 绕过：

```bash
export no_proxy="pypi.motphys.com"
export NO_PROXY="pypi.motphys.com"
make setup-motrix
```

同时设大小写是因为 `uv` 的 Rust HTTP 客户端两者都检查。

## httpx SOCKS 代理缺少 socksio

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

`huggingface_hub` 使用 `httpx`，检测到 `ALL_PROXY=socks5://...` 后需要
`socksio`。安装到项目 `.venv/`（不是 conda）：

```bash
uv pip install httpx[socks] --python .venv/bin/python
```

或改用 HTTP 代理避开 SOCKS：

```bash
unset all_proxy ALL_PROXY
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
```

## SAC / TD3 训练立即退出（exit code 247）

```
resource_tracker: There appear to be 5 leaked semaphore objects to clean up at shutdown
```

Off-policy 算法在 collector subprocess 中 JIT 编译 C++ CUDA 扩展
`unilab_native_h2d`。编译失败时 subprocess 静默退出，无 traceback。
用 `--sim mujoco` 跑同一任务可以看到完整报错。

编译需要三个前置条件：

**缺 C++ 编译器** — `c++: not found`：

```bash
sudo apt-get install build-essential
```

**缺 CUDA Toolkit 头文件** — `cuda_runtime_api.h: 没有那个文件`：

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

PyTorch pip wheel 只含运行时 `.so`，不含编译头文件。PPO/APPO 不需要
Toolkit，SAC/TD3 的 JIT 编译需要。

**conda CUDA Toolkit 头文件路径不标准**：

```bash
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
```

conda 把头文件放在 `targets/x86_64-linux/include/` 而非
`$CUDA_HOME/include/`。设好后 JIT 结果缓存到
`~/.cache/torch_extensions/`，后续不再编译。

## conda install 连接失败

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

conda 不读 shell 的 `http_proxy`。单独配置：

```bash
conda config --set proxy_servers.http http://127.0.0.1:7897
conda config --set proxy_servers.https http://127.0.0.1:7897
```

## ffmpeg 缺失

```
RuntimeError: Program 'ffmpeg' is not found
```

训练完成后回放录制视频需要 ffmpeg：

```bash
sudo apt install ffmpeg
```

## 持久化环境变量

将以下内容添加到 `~/.bashrc`：

```bash
# 代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy="pypi.motphys.com,localhost,127.0.0.1"
export NO_PROXY="pypi.motphys.com,localhost,127.0.0.1"

# CUDA Toolkit（conda 路径，按实际修改）
export CUDA_HOME=/home/<user>/anaconda3/envs/unilab
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 速查表

| 现象 | 解决 |
|------|------|
| motrixsim-core 超时 | `no_proxy=pypi.motphys.com` |
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| SAC/TD3 立即退出 (247) | 用 `--sim mujoco` 看完整报错 |
| `c++: not found` | `sudo apt install build-essential` |
| `cuda_runtime_api.h` 缺失 | `conda install -c nvidia cuda-toolkit=12.8` |
| CUDA headers 路径不对 | `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| conda HTTP 000 | `conda config --set proxy_servers.https http://127.0.0.1:7897` |
| ffmpeg not found | `sudo apt install ffmpeg` |
