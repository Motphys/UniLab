# 扩展 UniLab：新地形

地形生成是一项冷路径的场景特性。请把生成与资源 materialization 保持在
`step()`、`reset()` 以及热路径域随机化循环之外。

## 实现清单

1. 如果地形需要新的 heightfield 几何，在
   `src/unilab/terrains/heightfield_terrains.py` 中添加一个
   `SubTerrainCfg` 实现。
2. 如果地形应当可以按名称选择，在 `src/unilab/terrains/config.py` 中用
   `@terrain_preset` 添加一个 preset。
3. 把地形接入 `TerrainGeneratorCfg`，可以通过已有的命名集合（例如
   `ROUGH_TERRAINS_CFG`），也可以通过 `conf/` 下的 owner YAML。
4. 通过 `src/unilab/base/scene.py` 中的 `SceneCfg.terrain` 与
   `TerrainSceneCfg` 把地形暴露给场景。
5. 如果某个 env 需要高度 observation，在 init 时通过
   `create_hfield_scanner(...)` 创建后端 scanner；通过返回的
   `BackendHeightScanner` 读取采样。
6. 让地形的 spawn、curriculum 与 observation 维度在归属的 env config 与
   `obs_groups_spec` 中得到体现。

## 在风险点附近验证

- 地形生成器形状与数值行为：
  `tests/terrains/test_terrain_generator.py`
- Rough locomotion 高度扫描与 spawn 行为：
  `tests/envs/locomotion/test_go2_rough_height_scan.py`、
  `tests/envs/locomotion/test_go2_terrain_spawn.py`、
  `tests/envs/locomotion/test_terrain_spawn.py`
- 后端 materialization 边界：`tests/utils/test_xml_utils.py`

## 仓库内证据

- 地形配置与 preset：`src/unilab/terrains/config.py`
- 地形生成器：`src/unilab/terrains/terrain_generator.py`
- Heightfield 地形类型：`src/unilab/terrains/heightfield_terrains.py`
- 高度扫描辅助工具：`src/unilab/envs/locomotion/common/height_scan.py`
