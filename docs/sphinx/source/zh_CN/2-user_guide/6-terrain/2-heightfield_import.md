# 高度场导入

高度场地形通过 `SceneCfg` 和地形生成器进行配置，然后在 init 路径上由后端实例化。核心测试使用生成式 heightfield 场景覆盖该契约。

## 需要阅读的文件

- `src/unilab/terrains/heightfield_terrains.py`
- `unisim.terrain.generator`
- `src/unilab/tasks/locomotion/common/height_scan.py`
- `tests/utils/test_xml_utils.py`
- `unisim.backend.mujoco.xml`
- `unisim.backend.motrix.scene`

## 冒烟命令

```bash
uv run pytest tests/utils/test_xml_utils.py \
  -k materialize_mujoco_hfield_attached_scene
```

高度扫描的 ID 和偏移在 env 初始化期间缓存；热路径调用后端高度 scanner contract，而不是解析 XML 或 asset 元数据。
