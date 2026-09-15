# 高度场导入

高度场地形通过 `SceneCfg` 和地形生成器进行配置，然后在 init 路径上由后端实例化。已提交的面向用户示例是保留的 `g1_walk_rough` 场景。

## 需要阅读的文件

- `src/unilab/terrains/heightfield_terrains.py`
- `unisim.terrain.generator`
- `src/unilab/assets/robots/g1/scene_rough.xml`
- `src/unilab/tasks/locomotion/common/height_scan.py`
- `unisim.backend.mujoco.xml`
- `unisim.backend.motrix.scene`

## 冒烟命令

```bash
uv run train --algo sac --task g1_walk_rough --sim mujoco \
  algo.max_iterations=2 \
  algo.num_envs=64 \
  training.no_play=true
```

高度扫描的 ID 和偏移在 env 初始化期间缓存；热路径调用后端高度 scanner contract，而不是解析 XML 或 asset 元数据。
