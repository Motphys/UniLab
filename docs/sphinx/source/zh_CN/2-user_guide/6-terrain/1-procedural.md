# 程序化地形

程序化地形是 scene composition 能力，不是独立的 backend 模式。task owner
在 Hydra 中声明 `env.scene.terrain`；UniLab 校验配置，UniSim 的公开 backend
adapter 只在环境构建阶段物化地形。

## Owner 声明

核心可复用部分包括：

- `unilab.base.scene.SceneCfg`：`terrain` 与 scene 文件。
- `unilab.terrains` sub-terrain factory：generator 组合。

例如，owner 可以声明生成式 terrain，而不让 backend 感知任务语义：

```yaml
env:
  scene:
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        _target_: unilab.terrains.TerrainGeneratorCfg
        seed: 42
```

UniLab 在 `src/unilab/base/terrain.py` 中持有运行时地形 spawn 状态；UniSim
持有 generator 组合与 backend materialization。启用该能力的 task owner 需要自行
持有训练证据。

## 物化边界

`registry.make(...)` 期间：

1. Hydra 将选中的 owner 物化为 `ManagerBasedRlEnvCfg`。
2. `create_backend` 解析所需机器人资产，并将 `SceneCfg` 传给选中的 UniSim
   backend。
3. backend 的公开 scene materializer 生成 terrain matrix、origins、hfield
   与合并后的 scene model。
4. task owner 自行持有 scanner binding 与缓存 ID。

step 和 reset 不解析机器人 XML，也不检查 asset metadata；它们只消费缓存
ID 和公开 backend capability。

## Owner 集成

选择声明了 `env.scene.terrain` 的 task owner，然后通过 Hydra 调整 terrain
网格与 seed：

```text
env.scene.terrain.generator.num_rows=4
env.scene.terrain.generator.seed=42
```

## 运行时课程

启用 `env.scene.terrain.generator.curriculum=true` 后，`env.scene.terrain`
暴露形状为 `(num_rows, num_cols, 3)` 的 `terrain_origins`、逐环境的
`terrain_levels` 与 `terrain_types`，以及选中的 `env_origins`。原点 tensor
原地更新，因此 root reset 和缓存的 command 引用无需重建地形几何即可看到新 tile。

在 owner YAML 中添加进度项，按训练进度开放地形行：

```yaml
curriculum:
  terrain_levels:
    func: unilab.envs.mdp.terrain_levels_by_progress
    params:
      steps: 10000
```

每次 reset 时，该项仅为正在 reset 的环境从已开放的行中采样等级，保持其地形
类型不变，并在 root-state reset 事件之前更新原点。`steps` 必须是正整数。配置
`unilab.envs.mdp.reset_root_state_uniform` 等 root reset 项，将机器人放置到选中的
原点。

## 核心验证

```bash
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q
```

核心验证同时覆盖 terrain generator 与 backend materialization 边界；
额外任务族的证据由 ecosystem 包持有。
