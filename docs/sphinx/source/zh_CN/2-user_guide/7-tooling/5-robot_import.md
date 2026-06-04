# 机器人导入

这页是所有新机器人资产接入 UniLab 的通用指引。

机器人导入本身只负责资产、模型描述、命名、控制接口和低频 materialization；不要负责
task/reward/env 语义。

需要新增任务类型、改 observation/reward、episode 逻辑或 env owner 语义时，参阅
{doc}`../../4-developer_guide/3-extending/1-new_task`。

## 资产目录 contract

`src/unilab/assets/robots/<robot_name>/` 的资产应遵循仓库格式：

- `assets/`：只放 `.stl` / `.obj` mesh。
- `<robot_name>.xml`：纯机器人 MJCF，只包含 body / joint / actuator / sensor 等机器人描述。
- task/scene fragment：例如 `scene.xml`，放场景、地面、任务传感器和 keyframe。

不要复制完整外部资源包。`<keyframe>` 属于 task/scene fragment，不能放进机器人 XML。

## 输入资产

优先准备 MuJoCo/MJCF `.xml`，按照上述 contract 进行复制。

如果只有 URDF，使用仓库自带脚本进行转换：

```bash
uv run unilab-import-robot <urdf_path> [robot_name]
```

```{important}
为了加快仿真速度并提升接触稳定性，强烈建议简化机器人碰撞体：不要直接使用高面数
visual mesh 作为 collision mesh，尽量把碰撞体简化为 box / capsule / sphere / cylinder
等几何体。
```

- 默认自动导入会把 actuator 写成 `position`，这只适合位置控制 owner。
  - 如果机器人必须保留 torque/motor actuator 语义，后续扩展任务时，需要参考
    `src/unilab/envs/locomotion/go2w/` 的控制方式，把 action 解释、PD/力矩控制和
    actuator contract 放在机器人 owner 的控制边界内。
- 转换完成后，会自动弹出 `mujoco.viewer` 可视化界面展示转换结果，并进行下一步调整
  Keyframe。

## 调整 Keyframe

`home` keyframe 是接入者必须手动确认的机器人起始姿态。

- 在上一步“输入资产”所运行的脚本最后弹出的 viewer 右侧面板里拖动 `ctrl`，找到合适初始姿态（`home` keyframe）。
- 为了方便调高度，工具默认会在临时 viewer 模型里把 freejoint 展开成可调的 xyz/姿态关节，
  并额外加一个高度 position actuator；拖右侧 control 面板里的高度 slider 即可调 base z。
- 关闭 viewer 后，keyframe 自动写入 `scene.xml`。

检查 `home` 时至少确认：

- floating base 的高度让足端正常接触地面，而不是悬空或穿地。可以按下键盘 `c` 显示接触点，
  便于判断是否和地面接触。
- 关节角在 joint range 内，且接近自然站姿或任务起始姿态。

## 输出产物

使用 `uv run unilab-import-robot <urdf_path> [robot_name]` 转换后，会在仓库内生成：

- `src/unilab/assets/robots/<robot_name>/assets/`：转换并整理后的 mesh 资产。
- `src/unilab/assets/robots/<robot_name>/<robot_name>.xml`：机器人 MJCF 描述，只包含机器人
  本体、关节、actuator、sensor 和 mesh 引用。
- `src/unilab/assets/robots/<robot_name>/scene.xml`：task/scene fragment，包含 `home`
  keyframe。关闭 viewer 后，调好的 `home` 会写回这里。

这些产物只完成机器人资产导入，不会自动生成 task/reward/env owner。后续任务接入应按任务
自身文档扩展。
