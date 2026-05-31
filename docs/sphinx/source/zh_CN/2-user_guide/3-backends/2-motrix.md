# Motrix 后端

Motrix 是一个可选后端，通过 `motrix` extra 安装。在 `pyproject.toml` 中固定的
软件包是 `motrixsim-core==0.8.1.dev104665`，适配层位于
`src/unilab/base/backend/motrix/` 下。

## 安装

```bash
uv sync --extra motrix
```

`make setup-motrix` 会执行相同的依赖同步，并安装 shell 自动补全。

## 何时使用

- task owner 以 `conf/.../<task>/motrix.yaml` 形式存在。
- 你想要 Motrix 原生交互式回放；该后端提供原生交互式渲染器和视频录制能力。
- 所生成的支持矩阵将你的 entrypoint/task/backend 组合标记为 configured 或 tested。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

使用 `--render-mode record` 进行无头的仅视频回放。后端选择请保留在
`--sim motrix` 中，而不要单独 override `training.sim_backend`。
