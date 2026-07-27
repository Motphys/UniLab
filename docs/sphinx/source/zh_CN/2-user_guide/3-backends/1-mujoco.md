# MuJoCo 后端

MuJoCo 是已提交 owner 配置中的默认后端路径。其 Python 依赖为官方
`mujoco` 包（`>=3.5,<3.11`）加 `mujoco-uni-runtime`（见 `pyproject.toml`），适配层位于
`src/unilab/base/backend/mujoco/` 下。

## 何时使用

- 你想要 PPO、APPO、off-policy SAC/TD3 或 FlashSAC 的默认训练路线。
- task owner 仅以 `conf/.../<task>/mujoco.yaml` 形式存在。
- 你需要 MuJoCo 专有工具，例如 `scripts/play_viser.py`，或从 MuJoCo XML/MJB
  模型导出场景。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

回放模式由 `src/unilab/base/backend/base.py` 中的 backend contract 解析。
MuJoCo 在 `src/unilab/base/backend/mujoco/backend.py` 中声明对物理状态回放的支持；
`auto` 回放会录制视频，而不是打开 Motrix 原生交互式渲染器。

## 切换 MuJoCo 版本

mujoco extra 支持 `>=3.5,<3.11` 窗口内的任意求解器版本。全新安装默认使用
已提交的 `uv.lock` 中钉住的版本（当前为 **3.8.0**）。
`mujoco-uni-runtime` 的原生扩展针对本环境中的 `mujoco` 编译，且拒绝在其它版本下加载，
因此切换版本 = 重钉 `mujoco` + 重编扩展：

```bash
make mujoco MJ=3.8.0
```

该目标依次执行 `uv lock --upgrade-package mujoco==3.8.0`、清除 uv 对
`mujoco-uni-runtime` 的构建缓存（缓存无法感知扩展对 mujoco 版本的依赖）、
并强制在本环境内重新编译同步。不用 Makefile 时的等价命令：

```bash
uv lock --upgrade-package mujoco==3.8.0
uv cache clean mujoco-uni-runtime
uv sync --extra mujoco --extra motrix --reinstall-package mujoco-uni-runtime
```

如果省略清缓存或强制重装，uv 可能复用按旧版本 mujoco 编译的扩展，
import 时会以动态链接器错误失败（fail-closed，不会静默出错行为）。
