# MuJoCo 后端

MuJoCo 是已提交 owner 配置中的默认后端路径。其 Python 依赖为官方
`mujoco` 包（`~=3.11.0`，默认版本由已提交的 `uv.lock` 精确钉住）加
`mjbatch` 原生 batch 引擎（当前在 `pyproject.toml` 中钉住
[集成 fork](https://github.com/unilabsim/mjbatch)），适配层位于
`unisim.backend.mujoco` 下。

## 何时使用

- 你想要 PPO、APPO、off-policy SAC/TD3 或 FlashSAC 的默认训练路线。
- task owner 仅以 `src/unilab/conf/.../<task>/mujoco.yaml` 形式存在。
- 你需要 MuJoCo 专有工具，例如 `scripts/play_viser.py`，或从 MuJoCo XML/MJB
  模型导出场景。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

回放模式由 `unisim.backend.base` 中的 backend contract 解析。
MuJoCo 在 `unisim.backend.mujoco.backend` 中声明对物理状态回放的支持；
`auto` 回放会录制视频，而不是打开 Motrix 原生交互式渲染器。

## 切换 MuJoCo 版本

pyproject 约束 `mujoco~=3.11.0`；默认版本由已提交的 `uv.lock` 精确钉住，
uv 的 prefer-locked 语义保证普通 relock 不会漂移。`mjbatch` 引擎针对
`mujoco==3.11.0` 构建，并记录编译时的 `mujoco` 版本，拒绝在其它版本下导入
（fail-closed，不会静默出错行为）。因此切换版本需要对应版本的 `mjbatch`
构建：

1. 在 `pyproject.toml`（并同步镜像 `pyproject.rocm.toml`）中提升 `mujoco`
   边界和 `mjbatch` 源码钉版；
2. 重新锁定（`uv lock`，ROCm lockfile 通过 `make sync-rocm`）并重新同步
   （`uv sync --extra mujoco`）。

fork 的构建在编译期钉住 `mujoco==3.11.0`，因此隔离构建总是针对匹配的 mujoco
编译。引擎当前从钉住的集成 fork 消费；其最终分发身份（PyPI package 还是
git 钉版、是否提供预编译 wheel）是 roadmap 上的待定事项——在此之前，版本提升
需要与 [fork](https://github.com/unilabsim/mjbatch) 维护者协调。
