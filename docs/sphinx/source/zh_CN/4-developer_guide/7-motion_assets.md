# Motion 资产迁移指南（Hugging Face）

## 背景

motion 资产（`.npz` / `.csv`）已从 Git 仓库迁移到 Hugging Face 数据集仓库
[unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)，
以降低仓库体积、改善 clone 和 CI 体验。

本地目录 `src/unilab/assets/motions/g1/` 保留，作为下载落盘位置，原有路径引用保持有效。

## 首次使用

1. 安装依赖（`huggingface_hub` 已包含在核心依赖中）：

   ```bash
   uv sync
   ```

2. 直接运行训练 / 评估命令，motion 文件会在 `MotionLoader` 初始化时按需下载：

   ```bash
   uv run train --algo ppo --task g1_motion_tracking --sim mujoco
   ```

   首次下载时日志会输出：

   ```
   INFO:unilab.assets.hub:Downloading motions/g1/dance1_subject2_part.npz from HF repo unilabsim/unilab-motions ...
   INFO:unilab.assets.hub:Downloaded to /path/to/src/unilab/assets/motions/g1/dance1_subject2_part.npz
   ```

3. 下载完成后文件缓存在本地，后续运行不再触发下载。

## 离线使用

设置环境变量禁止网络请求：

```bash
export HF_HUB_OFFLINE=1
```

此时 resolver 只查找本地文件，找不到则报错。

在有网络的环境中提前下载全部资产：

```bash
huggingface-cli download unilabsim/unilab-motions \
  --repo-type dataset \
  --local-dir src/unilab/assets
```

下载完成后即可在离线环境中正常使用。

## CI 缓存

在 CI 中可通过设置 `HF_HOME` 指向持久化缓存目录来避免重复下载：

```yaml
env:
  HF_HOME: /cache/huggingface
```

或使用 `--local-dir` 预下载到仓库内目录（已被 `.gitignore` 排除）。

## 新增 motion 文件

1. 按现有流程生成 `.npz`（见 `scripts/motion/README.md`）。
2. 上传到 HF 仓库，保持目录结构一致：

   ```bash
   huggingface-cli upload unilabsim/unilab-motions \
     src/unilab/assets/motions motions \
     --repo-type dataset
   ```

3. 在 env config 中引用新文件路径即可。

## 机器人网格资产

机器人二进制网格（`.STL`）采用相同方式外置，托管在 Hugging Face 数据集仓库
[unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots)。
X2 网格在首次使用时按需下载，落盘到原始路径 `src/unilab/assets/robots/x2/meshes/`，
因此 XML 的 `meshdir` 引用保持有效。无需运行任务即可提前预拉取：

```bash
uv run unilab-pull-assets --robot x2
```

新增某个机器人的网格：

1. 上传到 HF 仓库，保持目录结构一致：

   ```bash
   huggingface-cli upload unilabsim/unilab-robots \
     src/unilab/assets/robots/<robot>/meshes robots/<robot>/meshes \
     --repo-type dataset
   ```

2. 在 `.gitignore` 中忽略本地 `*.STL`（保留 `.gitkeep` 以维持目录）。
3. 在 env 的冷路径上调用一次目录 resolver，例如
   `resolve_robot_asset_dir("robots/<robot>/meshes", marker="<某>.STL")`。

## 架构说明

- 资产解析模块：`src/unilab/assets/hub.py`（`resolve_motion_files`）。
- 唯一集成点：`src/unilab/envs/motion_tracking/g1/motion_loader.py` 中的
  `MotionLoader.__init__`，在冷路径上调用一次 resolver。
- 热路径（`step` / `reset`）**不会**触发任何文件下载或解析。
- `ASSETS_ROOT_PATH` 定义不变，下载落盘位置与原始本地路径完全一致。
- 机器人网格使用同一目录 resolver（`resolve_robot_asset_dir`），集成点为
  `src/unilab/envs/motion_tracking/x2/flip_tracking.py` 中的
  `X2WallFlipTrackingEnv.__init__`，并通过 `unilab-pull-assets` CLI 暴露。
