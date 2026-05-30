# 高度场导入

语言: 简体中文

高度场地形通过 `SceneCfg` 和地形生成器进行配置，然后在 init 路径上由后端实例化。已提交的面向用户的示例是 `Go2JoystickRough`，其 owner 位于 `conf/ppo/task/go2_joystick_rough/mujoco.yaml` 和 `conf/ppo/task/go2_joystick_rough/motrix.yaml`。

## 需要阅读的文件

- `src/unilab/terrains/heightfield_terrains.py`
- `src/unilab/terrains/terrain_generator.py`
- `src/unilab/envs/locomotion/go2/rough.py`
- `src/unilab/base/backend/mujoco/xml.py`
- `src/unilab/base/backend/motrix/scene.py`

## 冒烟命令

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
  algo.max_iterations=2 \
  algo.num_envs=64 \
  training.no_play=true

uv run scripts/visualize_task_env.py --task Go2JoystickRough --backend mujoco --num_envs 4
```

高度扫描的 ID 和偏移在 env 初始化期间缓存；热路径调用后端高度 scanner contract，而不是解析 XML 或 asset 元数据。

## Navigation

- Index: [文档](0-index.md)
