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

## 机器人二进制资产

机器人二进制网格和纹理（例如 `.STL`、`.obj`、`.png`）采用相同方式外置，
托管在 Hugging Face 数据集仓库
[unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots)。
已注册的机器人为 a2、allegro_hand、g1、go2、go2_arm、microduck、sharpa_wave、
x2（见 `src/unilab/assets/hub.py` 的 `ROBOT_ASSET_SPECS`）。它们的
mesh/纹理目录在首次使用时按需下载，落盘到原始路径（例如 G1 的
`src/unilab/assets/robots/g1/assets/` 与 `robots/g1/textures/`），因此 XML 中的
原始相对路径保持有效。这些目录通过 `pyproject.toml` 的
`tool.uv.build-backend.source-exclude` 排除在 wheel/sdist 之外；pip 安装后首次
使用会下载到已安装的包目录内，之后离线复用该本地缓存。无需运行任务即可提前预拉取：

```bash
uv run unilab-pull-assets --robot g1
uv run unilab-pull-assets --robot x2
uv run unilab-pull-assets --robot all   # 所有已注册机器人
```

新增某个机器人的二进制资产：

1. 按目录上传到 HF 仓库，保持目录结构一致。一个机器人有多个资产目录时，
   每个目录分别上传。例如 G1：

   ```bash
   uv run hf upload unilabsim/unilab-robots \
     src/unilab/assets/robots/g1/assets robots/g1/assets \
     --repo-type dataset
   uv run hf upload unilabsim/unilab-robots \
     src/unilab/assets/robots/g1/textures robots/g1/textures \
     --repo-type dataset
   ```

2. 在 `.gitignore` 中忽略下载目录（整个目录托管到 HF 的机器人直接忽略整个目录；
   较早的 microduck 条目保留 `.gitkeep`），在
   `tool.uv.build-backend.source-exclude` 中排除该目录，并在 `ROBOT_ASSET_SPECS`
   中注册。
3. 经由 `create_backend` 构建的 scene 会被自动覆盖：
   `ensure_robot_assets_for_paths` 在 backend 解析 XML 之前的冷路径解析已注册目录。
   绕过 `create_backend` 的入口需要显式解析，例如 MicroDuck task factory：

   ```python
   resolve_robot_asset_dir("robots/microduck/assets", marker="trunk_base.stl")
   ```

## 架构说明

- 资产解析模块：`src/unilab/assets/hub.py`（`resolve_motion_files`）。
- motion 集成点：`src/unilab/tasks/motion_tracking/common/motion_loader.py` 中的
  `MotionLoader.__init__`，在冷路径上调用一次 resolver。
- 机器人 mesh 集成点：`src/unilab/base/backend_factory.py` 的 `create_backend`
  对 scene 的 `model_file`、`visual_model_file` 与 `fragment_files` 调用
  `ensure_robot_assets_for_paths`，然后再分发给具体 backend。
- 热路径（`step` / `reset`）**不会**触发任何文件下载或解析。
- `ASSETS_ROOT_PATH` 定义不变，下载落盘位置与原始本地路径完全一致。
- 机器人二进制资产使用同一目录 resolver（`resolve_robot_asset_dir`）。X2 和
  MicroDuck 的薄 task factory 都会先在冷路径解析一次，再委托给共享
  manager env factory；同一 resolver 也通过 `unilab-pull-assets` CLI 暴露。
