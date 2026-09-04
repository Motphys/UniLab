# Drake 后端

Drake 是实验性的 CPU 批量物理后端；task、reward、observation 和训练循环仍由
UniLab 负责。渲染使用 MuJoCo 原生 renderer：Drake 推进物理，MuJoCo 只绘制当前状态。
原生路径支持 Linux x86_64 与 Apple Silicon macOS（arm64）；Intel
macOS 没有官方 Drake 二进制。

## 前置依赖

- Python `>=3.10,<3.14` 和 `uv`。
- Linux：C++20 工具链及开发包：

  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential pkg-config libeigen3-dev libfmt-dev \
    libspdlog-dev curl git
  ```

- Apple Silicon macOS：Apple Clang 和 Homebrew 运行库：

  ```bash
  xcode-select --install  # 如果没有 clang++
  brew install fmt gcc
  ```

  `gcc` 提供 Drake macOS 构建所引用的 `libgfortran`。

## 安装

在 UniLab checkout 根目录运行：

```bash
make setup-drake
```

该目标会下载当前主机适配的 Drake 1.56.0 tarball，安装 `drake-uni` extra，使用
当前 uv Python 编译原生扩展，并运行 batch 诊断。流程可重复执行，文件和日志默认
保存在 `~/.unilab/drake`。

已有 Drake 安装可指定前缀：

```bash
make setup-drake DRAKE_HOME=/path/to/drake
```

前缀必须包含 `include/drake/`、`include/pybind11/` 和 `lib/libdrake.so`。macOS
下脚本会自动发现 Homebrew `fmt`，并把 gcc 库目录加入 `DYLD_LIBRARY_PATH`。

脚本结束时会打印后续 shell 需要的环境变量。Apple Silicon macOS 通常为：

```bash
export DRAKE_HOME="$HOME/.unilab/drake/drake-1.56.0-mac-arm64"
export UNILAB_DRAKE_HOME="$DRAKE_HOME"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current:$DRAKE_HOME/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

## 验证

在未导入 `pydrake` 的新进程中运行：

```bash
uv run --no-sync python - <<'PY'
from drake_uni.runtime import batch_diagnostics

diagnostics = batch_diagnostics()
print(diagnostics)
if not diagnostics.batch_available:
    raise SystemExit(diagnostics.batch_import_error or "Drake batch extension is unavailable")
PY
```

输出必须包含 `batch_available=True`。

## 训练 Go2

首次运行前预取 Go2 资产：

```bash
uv run --no-sync unilab-pull-assets --robot go2
```

使用 `--sim drake` 选择后端，不要手动覆盖 `training.sim_backend`。标准 PPO 命令为：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim drake
```

该命令使用 Drake owner 配置中的 1024 个环境、151 次迭代，并固定使用 CPU 训练
（Drake 暴露的是 float64 NumPy buffer）。Drake owner 使用场景 keyframe 重置；当前
backend contract 尚未提供浮动根状态随机化。只想验证安装时，可临时追加
`algo.max_iterations=1 algo.num_envs=4 algo.num_steps_per_env=4 training.no_play=true
env.drake_nthread=1`；这些不是生产参数。

在 Apple Silicon macOS 上已用 Drake 1.56.0、Python 3.13 和 1024 个环境完成该命令
的全部 151 次迭代，耗时约 254 秒。

Drake 没有独立 renderer。自动录制和交互 viewer 都使用 MuJoCo，实际推进的物理引擎
始终只有 Drake。无头评估使用 `--render-mode none`；录制和交互回放需要 MuJoCo
extra 与视觉资产。MuJoCo 不会推进物理，也不会重新评估 checkpoint。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: drake_uni` | 重新运行 `make setup-drake`。 |
| `DrakeEnvPool batch extension has not been built` | 使用相同的 uv Python 和 Drake 前缀重新编译，不能复制其他 ABI 的扩展。 |
| `libdrake.so` 或 `libgfortran` 无法加载 | 设置 `DRAKE_HOME`，Linux 使用 `LD_LIBRARY_PATH`，macOS 使用 `DYLD_LIBRARY_PATH`，并包含 Homebrew gcc 目录。 |
| Eigen/fmt 链接错误 | Linux 安装上述开发包；macOS 执行 `brew install fmt gcc`。 |
