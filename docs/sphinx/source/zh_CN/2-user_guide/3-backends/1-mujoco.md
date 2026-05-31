# MuJoCo 后端

MuJoCo 是已提交 owner 配置中的默认后端路径。其 Python 依赖在
`pyproject.toml` 中为 `mujoco-uni==3.8.0`，适配层位于
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
