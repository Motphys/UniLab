# 运动任务

UniLab 保留少量 reference/conformance 运动任务以及共享 Manager-Based
runtime。Unitree production 变体由下游 `unitree_rl_unilab` 包维护。

## 核心参考任务

- Go2 joystick: `go2_joystick_flat`
- G1 walking: `g1_walk_flat`
- G1 motion tracking 参考配置：见 {doc}`2-motion_tracking`

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

按 entrypoint、task owner 和 backend 查询核心支持矩阵：
{doc}`../../5-reference/5-support_matrix`。

## Unitree ecosystem 任务

Unitree production 变体维护在
[unitree_rl_unilab](https://github.com/unilabsim/unitree_rl_unilab)，该仓库只依赖
已发布的 UniLab/UniRL 发行版。其支持证据与任务文档由下游仓库持有，不属于
UniLab 核心支持矩阵。
