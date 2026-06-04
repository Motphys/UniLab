# FAQ

Common problems when installing and first running UniLab from a source checkout.
The examples below use generic proxy placeholders; adapt proxy hosts, ports, and
CUDA paths to your environment.

## httpx SOCKS proxy missing socksio

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

`huggingface_hub` uses `httpx`, which requires `socksio` once it detects
`ALL_PROXY=socks5://...`. Install it into the project `.venv/` (not conda):

```bash
uv pip install httpx[socks] --python .venv/bin/python
```

Or switch to an HTTP proxy to avoid SOCKS entirely:

```bash
unset all_proxy ALL_PROXY
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

Replace `proxy.example.com:8080` with your own proxy endpoint.

## Native H2D extension unavailable

```
Native H2D extension unavailable: CUDA_HOME not set and nvcc not found.
```

The off-policy CUDA replay pipeline may JIT-compile the optional
`unilab_native_h2d` extension. UniLab falls back to the pure-PyTorch async copy
path when the native extension is unavailable, so this message is usually
diagnostic rather than fatal. If you want the native extension path, check these
toolchain prerequisites:

**C++ compiler**:

```bash
sudo apt-get install build-essential
```

**CUDA Toolkit**:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

**CUDA path**:

```bash
export CUDA_HOME=$CONDA_PREFIX
```

UniLab checks both `$CUDA_HOME/include` and
`$CUDA_HOME/targets/x86_64-linux/include` for `cuda_runtime_api.h`.

## conda install connection fails

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

Some conda setups do not use shell proxy variables. Configure conda's own proxy
settings when package resolution fails behind a proxy:

```bash
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https http://proxy.example.com:8080
```

## ffmpeg missing

```
RuntimeError: Program 'ffmpeg' is not found
```

Replay video recording after training needs ffmpeg:

```bash
sudo apt install ffmpeg
```

## Quick reference

| Symptom | Fix |
|---------|-----|
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| native H2D extension unavailable | install C++ / CUDA Toolkit prerequisites or use the fallback |
| conda HTTP 000 | configure `conda config --set proxy_servers.*` for your proxy |
| ffmpeg not found | `sudo apt install ffmpeg` |
