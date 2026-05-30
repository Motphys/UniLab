# 运动控制

语言: 简体中文

运动控制任务注册在 `src/unilab/envs/locomotion/` 和
`src/unilab/envs/motion_tracking/` 中。`conf/` 下可用的 owner YAML
定义了哪些算法与后端组合是可运行的。

## 系列

- Go1：`go1_joystick_flat`、`go1_joystick_rough`
- Go2：`go2_joystick_flat`、`go2_joystick_rough`、`go2_handstand`、`go2_footstand`
- Go2W：`go2w_joystick_flat`、`go2w_joystick_rough`
- G1 行走：`g1_walk_flat`、`g1_walk_rough`
- G1 动作追踪：`g1_motion_tracking`、`g1_flip_tracking`、
  `g1_wall_flip_tracking`、`g1_climb_tracking`、`g1_box_tracking`
- Go2 机械臂：`go2_arm_manip_loco`

## 示例

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_rough --sim motrix training.no_play=true
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

查看支持矩阵以了解按 entrypoint、task owner 和 backend 划分的证据分级：
{doc}`../../5-reference/5-support_matrix`。

## Navigation

- Index: [文档](0-index.md)
