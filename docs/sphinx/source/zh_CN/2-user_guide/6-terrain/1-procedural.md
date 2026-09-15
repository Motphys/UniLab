# 程序化地形

程序化地形是 scene composition 能力，不是独立的 backend 模式。task owner
在 Hydra 中声明 `env.scene.terrain`；UniLab 校验配置，UniSim 的公开 backend
adapter 只在环境构建阶段物化地形。

## Owner 声明

核心可复用部分包括：

- `unilab.base.scene.SceneCfg`：`terrain` 与 scene 文件。
- `unilab.tasks.locomotion.common.rough_manager_terms`：terrain generator、
  rough reset、velocity command、height scan 与 action owner term。
- `unilab.tasks.locomotion.common.height_scan`：缓存式高度采样。

例如，owner 可以声明生成式 terrain，而不让 backend 感知任务语义：

```yaml
env:
  scene:
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        _target_: unilab.tasks.locomotion.common.rough_manager_terms.QuadrupedRoughTerrainCfg
        seed: 42
```

UniLab 不保留 procedural-terrain production owner。generator 组合与 backend
materialization 由核心 terrain 和 materialization 测试覆盖；启用该能力的
task owner 需要自行持有训练证据。

## 物化边界

`registry.make(...)` 期间：

1. Hydra 将选中的 owner 物化为 `ManagerBasedRlEnvCfg`。
2. `create_backend` 解析所需机器人资产，并将 `SceneCfg` 传给选中的 UniSim
   backend。
3. backend 的公开 scene materializer 生成 terrain matrix、origins、hfield
   与合并后的 scene model。
4. entity 与 height-scanner ID 被缓存供 reset/step 使用。

step 和 reset 不解析机器人 XML，也不检查 asset metadata；它们只消费缓存
ID 和公开 backend capability。

## Owner 集成

选择声明了 `env.scene.terrain` 的 task owner，然后通过 Hydra 调整 terrain
网格与 seed：

```text
env.scene.terrain.generator.num_rows=4
env.scene.terrain.generator.seed=42
```

## 核心验证

```bash
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q
```

核心验证同时覆盖 terrain generator 与 backend materialization 边界；
额外任务族的证据由 ecosystem 包持有。
