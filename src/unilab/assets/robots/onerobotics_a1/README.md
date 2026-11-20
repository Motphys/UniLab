# OneRobotics A1 Robot Simulation Assets

## Overview

本目录包含用于 UniLab 的 OneRobotics A1 机器人资产，按独立 CAD 模型分为：

- `A1 Left Arm`：左臂；
- `A1 Right Arm`：右臂；
- `A1 Bimanual Stand`：双臂站架。

三个模型是相互独立的资产，不应在运行时拼接为同一个模型。当前版本不包含
夹爪、传感器、控制器或训练策略。

## Contents

- **urdf**：三套经过便携化处理的 URDF；
- **meshes**：按模型分开的 STL 网格；
- **mjcf**：三套可由 UniLab MuJoCo demo 直接加载的 MJCF；
- **LICENSE**：OneRobotics A1 资产的 CC BY 4.0 完整许可文本；
- **ASSET_LICENSE_STATUS.md**：受许可文件清单、署名和范围说明。

## Models

| 模型 | URDF | MJCF | Demo |
|---|---|---|---|
| 左臂 | `urdf/a1_left.urdf` | `mjcf/a1_left.xml` | `a1-left-arm` |
| 右臂 | `urdf/a1_right.urdf` | `mjcf/a1_right.xml` | `a1-right-arm` |
| 双臂站架 | `urdf/a1_bimanual_stand.urdf` | `mjcf/a1_bimanual_stand.xml` | `a1-bimanual` |

三套 URDF 均使用包内相对 mesh 路径，不依赖当前资产目录以外的文件。

## Demos

安装 MuJoCo 可选依赖后，可运行三个无需 checkpoint 的关节运动 demo：

```bash
uv sync --extra mujoco
uv run demo a1-left-arm
uv run demo a1-right-arm
uv run demo a1-bimanual
```

demo 以运动学方式驱动关节，在交互式查看器中展示资产加载、关节范围和模型结构。

## Sources

- Original repository: <https://github.com/katazen/onerobot_h1>
- Source revision: `ecf530911284ba0e559f7a24dc222fd8e60d31ed`
- Packaged URDF files: `urdf/`
- Packaged mesh files: `meshes/`

规范化源文件全部位于当前 `onerobotics_a1` 目录内：

- 左臂源模型：`urdf/a1_left.urdf`、`meshes/a1_left/`；
- 右臂源模型：`urdf/a1_right.urdf`、`meshes/a1_right/`；
- 双臂站架源模型：`urdf/a1_bimanual_stand.urdf`、
  `meshes/a1_bimanual_stand/`。

## Validation

| 模型 | Links | Joints | Mesh references |
|---|---:|---:|---:|
| 左臂 | 9 | 8 | 16 |
| 右臂 | 8 | 7 | 16 |
| 双臂站架 | 17 | 16 | 34 |

三套 URDF 的所有 mesh 引用均可在当前资产目录内解析。

## License

OneRobotics A1 robot assets © 2026 OneRobotics，采用 Creative Commons
Attribution 4.0 International（CC BY 4.0）。完整条款见 [`LICENSE`](LICENSE)，
详细覆盖范围及署名要求见 [`ASSET_LICENSE_STATUS.md`](ASSET_LICENSE_STATUS.md)。

修改后的再分发必须保留版权与许可信息、注明 OneRobotics 为创作者，并说明所做
修改。许可不授予 OneRobotics 商标权，也不得暗示官方背书。

## Changelog

- 2026-10-13：加入左臂、右臂和双臂站架的 MJCF 与交互 demo。
- 2026-10-12：加入左臂、右臂、双臂站架 URDF 和 STL 资产。
