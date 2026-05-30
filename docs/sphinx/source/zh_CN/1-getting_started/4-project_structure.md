# 项目结构

语言: 简体中文

UniLab 将运行时 contract、配置、训练脚本和文档分置于不同的 owner 区域。当你需要在改动行为之前找到正确的层时，请使用这张索引图。

| 路径 | Owner 角色 |
| --- | --- |
| `scripts/` | 轻量的训练与工具入口。脚本负责组合 Hydra 配置并调用 owner 层代码。 |
| `conf/` | Hydra 根配置和任务 owner YAML。顶层 CLI 将后端选择暴露为 `--task` 加 `--sim`，然后组合出匹配的 owner YAML。 |
| `src/unilab/base/` | Registry、env state、scene 以及 backend contract。 |
| `src/unilab/envs/` | 任务 env 实现，以及任务专属的 reset、reward、observation 和 DR 逻辑。 |
| `src/unilab/algos/` | PPO、APPO、off-policy、MLX、HIM-PPO 和 HORA 算法代码。 |
| `src/unilab/ipc/` | 共享内存与异步 runner 原语。 |
| `src/unilab/training/` | 共享的训练辅助工具，用于日志、回放、种子处理和配置守卫（config guard）。 |
| `src/unilab/visualization/` | 回放、渲染、NaN 检查以及 scene/export 工具。 |
| `tests/` | Contract、config、env、algorithm、script 和集成测试。 |
| `docs/sphinx/source/en/` | 英文的用户、部署、开发者和参考文档。 |
| `docs/sphinx/source/zh_CN/` | 中文文档，其兼容性路径由语言切换器处理。 |

## 配置布局

主要的配置根为：

- `conf/ppo/config.yaml`，用于 torch PPO。
- `conf/ppo/config_mlx.yaml`，用于 MLX PPO。
- `conf/appo/config.yaml`，用于 APPO。
- `conf/offpolicy/config.yaml` 加上 `conf/offpolicy/algo/*.yaml`，用于 SAC、
  TD3 和 FlashSAC。
- `conf/ppo_him/config.yaml` 和 `conf/hora_distill/config.yaml`，用于
  专门的 HIM-PPO 和 HORA 路径。

任务 owner YAML 即后端身份。示例：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

不要仅通过覆盖 `training.sim_backend` 来切换后端。

## 后续去向

- 用户训练命令：{doc}`../2-user_guide/1-training/1-cli_reference`
- Hydra owner YAML：{doc}`../2-user_guide/1-training/2-hydra_config`
- 面向贡献者的 contract：{doc}`../4-developer_guide/0-index`

## Navigation

- Index: [文档](0-index.md)
