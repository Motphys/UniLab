# 统一 CLI

语言: 简体中文

`uv run train`、`uv run eval` 和 `uv run demo` 是 UniLab 的默认训练接口。

## train

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo td3 --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo mlx_ppo --task go2_joystick_flat --sim mujoco
```

支持的算法：`ppo`、`mlx_ppo`、`appo`、`sac`、`td3`、`flashsac`

支持的后端：`mujoco`、`motrix`

## Tab 自动补全

补全脚本是可选项，只补全 `uv run train` 和 `uv run eval` 的入口、flags 和部分 choices，不改变命令行为。新环境可用一条 setup 命令完成同步和补全安装：

```bash
make setup

# 需要 Motrix 时：
make setup-motrix
```

`make setup` 会执行 `uv sync` 和 `uv run --no-sync unilab-complete install`；`make setup-motrix` 会执行 `uv sync --extra motrix` 和同样的补全安装。安装命令会按 `$SHELL` / 平台选择 Bash 或 Zsh，只写入用户级 rc 文件。当前终端不会被子进程自动激活，重新打开终端或 source 对应 rc 文件后生效。

`make setup` / `make setup-motrix` 要求本机已安装 `make`。如果系统没有 `make`，可直接执行 `uv sync && uv run --no-sync unilab-complete install`，或需要 Motrix 时执行 `uv sync --extra motrix && uv run --no-sync unilab-complete install`。

Linux / WSL Bash 用户也可手动把下面内容写入 `~/.bashrc`：

```bash
if [ -f "/path_to_unilab/UniLab/scripts/completions/unilab.bash" ]; then
    source "/path_to_unilab/UniLab/scripts/completions/unilab.bash"
fi
```

macOS 默认 Zsh 用户可把下面内容写入 `~/.zshrc`：

```zsh
autoload -Uz compinit
compinit
source "/path_to_unilab/UniLab/scripts/completions/unilab.zsh"
```

重新打开终端或 source 对应 rc 文件后，可用 `uv run <TAB>`、`uv run train --algo <TAB>`、`uv run train --sim <TAB>` 查看候选项。把 `/path_to_unilab/UniLab` 替换成你的 UniLab checkout 路径。

## eval

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

## demo

```bash
uv run demo
uv run demo --preset go2_joystick_mujoco_ppo
uv run demo --refresh
uv run demo --device cpu
```

`demo` 会从本地预设 run 复制 checkpoint，再走回放命令；不是远端模型下载器。

## 规则

- `--algo`、`--task`、`--sim` 保持显式
- backend 选择走 `--sim`，不要单独切 `training.sim_backend`
- ROCm 环境在 `make sync-rocm` 后使用 `uv run ...`；Intel XPU 环境使用 `uv run --no-sync ...`
- `eval --load-run` 只接受 `-1` 或 run 目录名，不接受绝对路径

## render mode

- `auto`：MuJoCo 默认导出视频；Motrix 默认打开窗口
- `record`：只录视频
- `none`：不回放
- `interactive`：主要用于 Motrix 交互式回放

## Navigation

- Index: [Documentation](../../index.md)
- Previous: [训练指南](../03-training.md)
- Next: [评估、回放与恢复训练](02-playback-and-resume.md)
