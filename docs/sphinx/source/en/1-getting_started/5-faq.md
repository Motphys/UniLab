# FAQ

Common problems when installing and first running UniLab, organized as
symptom -> fix. Most relate to proxy environments (HTTP / SOCKS5), the JIT
compilation dependencies of off-policy algorithms, or video export.

## motrixsim-core download times out

```
× Failed to download `motrixsim-core==0.8.1.dev104665`
╰─▶ operation timed out
```

`pypi.motphys.com` is hosted in mainland China and is often unreachable through
an overseas proxy. Bypass it with `no_proxy`:

```bash
export no_proxy="pypi.motphys.com"
export NO_PROXY="pypi.motphys.com"
make setup-motrix
```

Set both cases because `uv`'s Rust HTTP client checks both variable names.

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
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
```

## SAC / TD3 training exits immediately (exit code 247)

```
resource_tracker: There appear to be 5 leaked semaphore objects to clean up at shutdown
```

Off-policy algorithms JIT-compile the C++ CUDA extension `unilab_native_h2d`
inside the collector subprocess. When compilation fails the subprocess exits
silently with no traceback. Run the same task with `--sim mujoco` to see the
full error.

Compilation needs three prerequisites:

**Missing C++ compiler** — `c++: not found`:

```bash
sudo apt-get install build-essential
```

**Missing CUDA Toolkit headers** — `cuda_runtime_api.h: No such file`:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

The PyTorch pip wheel ships only the runtime `.so`, not the compile headers.
PPO/APPO do not need the Toolkit; the SAC/TD3 JIT compilation does.

**conda CUDA Toolkit header path is non-standard**:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
```

conda places headers under `targets/x86_64-linux/include/` instead of
`$CUDA_HOME/include/`. Once set, the JIT result is cached to
`~/.cache/torch_extensions/` and is not recompiled on later runs.

## conda install connection fails

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

conda does not read the shell's `http_proxy`. Configure it separately:

```bash
conda config --set proxy_servers.http http://127.0.0.1:7897
conda config --set proxy_servers.https http://127.0.0.1:7897
```

## ffmpeg missing

```
RuntimeError: Program 'ffmpeg' is not found
```

Replay video recording after training needs ffmpeg:

```bash
sudo apt install ffmpeg
```

## Persistent environment variables

Add the following to `~/.bashrc`:

```bash
# Proxy
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export no_proxy="pypi.motphys.com,localhost,127.0.0.1"
export NO_PROXY="pypi.motphys.com,localhost,127.0.0.1"

# CUDA Toolkit (conda path, adjust to your install)
export CUDA_HOME=/home/<user>/anaconda3/envs/unilab
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include

# HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## Quick reference

| Symptom | Fix |
|---------|-----|
| motrixsim-core timeout | `no_proxy=pypi.motphys.com` |
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| SAC/TD3 exits immediately (247) | run with `--sim mujoco` to see the full error |
| `c++: not found` | `sudo apt install build-essential` |
| `cuda_runtime_api.h` missing | `conda install -c nvidia cuda-toolkit=12.8` |
| CUDA headers path wrong | `export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| conda HTTP 000 | `conda config --set proxy_servers.https http://127.0.0.1:7897` |
| ffmpeg not found | `sudo apt install ffmpeg` |
