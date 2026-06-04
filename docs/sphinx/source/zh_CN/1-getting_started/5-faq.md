# 常见问题

从源码 checkout 安装和首次运行 UniLab 时的常见问题。下面的示例使用通用代理占位
符；请按你的环境替换代理地址、端口和 CUDA 路径。

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
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

请将 `proxy.example.com:8080` 替换为你的代理入口。

## native H2D 扩展不可用

```
Native H2D extension unavailable: CUDA_HOME not set and nvcc not found.
```

off-policy CUDA replay pipeline 可能会 JIT 编译可选的 `unilab_native_h2d`
扩展。该扩展不可用时，UniLab 会回退到纯 PyTorch async copy 路径，因此这条信息通
常是诊断信息而不是致命错误。如果你希望启用 native extension 路径，请检查这些工
具链前置条件：

**C++ 编译器**：

```bash
sudo apt-get install build-essential
```

**CUDA Toolkit**：

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

**CUDA 路径**：

```bash
export CUDA_HOME=$CONDA_PREFIX
```

UniLab 会同时检查 `$CUDA_HOME/include` 和
`$CUDA_HOME/targets/x86_64-linux/include` 中的 `cuda_runtime_api.h`。

## conda install 连接失败

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

部分 conda 环境不会使用 shell proxy 变量。位于代理后方且解析包失败时，单独配置
conda 代理：

```bash
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https http://proxy.example.com:8080
```

## ffmpeg 缺失

```
RuntimeError: Program 'ffmpeg' is not found
```

训练完成后回放录制视频需要 ffmpeg：

```bash
sudo apt install ffmpeg
```

## 速查表

| 现象 | 解决 |
|------|------|
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| native H2D 扩展不可用 | 安装 C++ / CUDA Toolkit 依赖，或使用 fallback |
| conda HTTP 000 | 按你的代理配置 `conda config --set proxy_servers.*` |
| ffmpeg not found | `sudo apt install ffmpeg` |
