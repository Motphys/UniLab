# Motion 资产迁移指南（HF）

语言：简体中文

## 背景

motion 资产（`.npz` / `.csv`）已从 Git 仓库迁移到 Hugging Face 数据集仓库
[unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)，
以降低仓库体积、改善 clone 和 CI 体验。

本地目录 `src/unilab/assets/motions/g1/` 保留，作为下载落盘位置。

## 首次使用

1. 安装 `huggingface_hub`（可选依赖）：

   ```bash
   uv sync --extra hf
   # 或
   uv pip install huggingface_hub
   ```

2. 直接运行训练 / 评估命令，motion 文件会在 `MotionLoader` 初始化时自动下载：

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

### 预下载

在有网络的环境中提前下载全部资产：

```bash
huggingface-cli download unilabsim/unilab-motions \
  --repo-type dataset \
  --local-dir src/unilab/assets
```

下载完成后可在离线环境中正常使用。

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

## 历史覆写后协作者同步

Git 历史覆写（`git filter-repo`）完成后，所有协作者需要重新同步：

```bash
# 方案 A：如果没有本地未推送提交
git fetch origin
git reset --hard origin/main

# 方案 B：如果有本地工作
git fetch origin
git rebase origin/main
```

**注意**：历史覆写后，旧的 commit hash 将全部失效。请确保在覆写前推送所有本地提交。

## 架构说明

- 资产解析模块：`src/unilab/assets/hub.py`
- 唯一集成点：`MotionLoader.__init__`（冷路径）
- 热路径（`step` / `reset`）**不会**触发任何文件下载或解析
- `ASSETS_ROOT_PATH` 定义不变，下载落盘位置与原始本地路径完全一致
