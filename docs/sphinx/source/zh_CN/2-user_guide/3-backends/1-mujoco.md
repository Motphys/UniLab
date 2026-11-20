# MuJoCo 后端

MuJoCo 是已提交 owner 配置中的默认后端路径。在 Python 3.11 及更高版本上，
`mujoco` extra 安装官方 `mujoco` 包（`~=4.0.1`）和发布版
`mjbatch-uni==0.3.1` 原生 batch 引擎，精确版本由已提交的 `uv.lock` 钉住。
适配层位于 `unisim.backend.mujoco` 下。

## 何时使用

- 你想要 PPO、APPO、off-policy SAC 或 FlashSAC 的默认训练路线。
- task owner 仅以 `src/unilab/conf/.../<task>/mujoco.yaml` 形式存在。
- 你需要 MuJoCo 专有工具，例如 `scripts/play_viser.py`，或从 MuJoCo XML/MJB
  模型导出场景。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go2_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

回放模式由 `unisim.backend.base` 中的 backend contract 解析。
MuJoCo 在 `unisim.backend.mujoco.backend` 中声明对物理状态回放的支持；
`auto` 回放会录制视频，而不是打开 Motrix 原生交互式渲染器。

## 切换 MuJoCo 版本

在 Python 3.11 及更高版本上，pyproject 约束 `mujoco~=4.0.1` 并钉定
`mjbatch-uni==0.3.1`；默认精确版本由已提交的 `uv.lock` 钉住，uv 的
prefer-locked 语义保证普通 relock 不会漂移。`mjbatch` 引擎记录编译时的
`mujoco` 版本，拒绝在其它版本下导入（fail-closed，不会静默改变行为）。
因此切换版本需要兼容的 `mjbatch` 发布版本：

1. 在 `pyproject.toml`（并同步镜像 `pyproject.rocm.toml`）中提升 `mujoco`
   边界和 `mjbatch-uni` 钉版；
2. 重新锁定（`uv lock`，ROCm lockfile 通过 `make sync-rocm`）并重新同步
   （`uv sync --extra mujoco`）。

默认安装路径使用预编译 `mjbatch-uni` wheel。新的 MuJoCo/引擎版本组合需要与
[集成 fork](https://github.com/unilabsim/mjbatch) 协调。
