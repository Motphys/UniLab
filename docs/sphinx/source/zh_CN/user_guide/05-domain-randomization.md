# 域随机化

语言: 简体中文

## 三类生命周期

- init-lifecycle：改变模型身份或几何，只能发生在初始化 / materialization
- reset-lifecycle：reset 时下发参数或状态随机化
- interval-lifecycle：step 间外部扰动

## 用户最常碰到的 DR 项

- `base_mass_delta`、`base_com_offset`：常见于 locomotion
- `kp` / `kd`：常见于 Go2、G1 等 position-actuator 路径
- `gravity`：当前只看 MuJoCo 能力和任务 owner 是否启用
- `push_robots` / `body_force`：属于 interval-lifecycle
- `geom_size`：属于 init-lifecycle，不是 reset 时热更新字段

## 当前用户最常见的边界

- `gravity` 不是 Motrix 的默认 capability
- backend capability 不等于该任务默认启用了这项 DR
- `geom_size` 不属于 `ResetRandomizationPayload`
- 任务创建后再改 cold-path 字段，不会回写到已 materialize 的 env

## 常见命令例子

只放开 gravity 大小、保持竖直向下：

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

打开 interval push：

```bash
uv run scripts/train_rsl_rl.py \
  task=g1_walk_flat/mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]'
```

## 相关任务

- [G1 Motion Tracking](D-tasks/02-g1-motion-tracking.md)：训练前先确认 motion 资产和 replay
- [Sharpa Inhand](D-tasks/04-sharpa-inhand.md)：scale / grasp cache / DR 边界更敏感
- [Go2 Rough Terrain](D-tasks/05-go2-rough-terrain.md)：常见的是 mass、COM、friction、push

## 深入文档

- 需要更底层 contract 时，去看 developer 文档里的 [Domain Randomization Contract](../../developer_guide/zh_CN/domain-randomization-contract.md)

## Navigation

- Index: [Documentation](../index.md)
- Previous: [Algorithms](04-algorithms.md)
- Next: [任务索引](D-tasks/01-task-index.md)
