# UniLab

## 安装 (Installation)

1. **克隆仓库**:
   ```bash
   git clone https://github.com/TATP-233/UniLab.git
   cd UniLab
   ```

2. **安装依赖**:
   在项目根目录下运行：
   ```bash
   pip install -e .
   ```

## 训练与回放指南

### 1. 开始训练 (Training)
默认使用 `Go2JoystickFlatTerrain` 任务：

```bash
# 基本训练
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain

# 指定环境数量 (默认 1024)
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --env_num 2048
```

### 2. 回放与渲染视频 (Play / Evaluation)
增加 `--play_only` 参数。脚本默认会加载最新的一次 `run`，并在该次 run 的目录中生成 `play_video.mp4`。

```bash
# 加载最新的一次训练结果进行回放并渲染
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --play_only --load_run -1
```

### 3. 加载特定 Run 继续训练
如果你想从某个特定的检查点继续训练：

```bash
# --load_run 可以是 logs/rsl_rl_train/TaskName 下的文件夹名
python scripts/train_rsl_rl.py --task Go2JoystickFlatTerrain --load_run "2024-02-04_12-00-00"
```

### 参数说明
*   `--task`: 任务名称（如 `Go2JoystickFlatTerrain` 或 `Go1JoystickFlatTerrain`）。
*   `--play_only`: 仅推理回放模式，不进行训练，会生成并行渲染的视频。
*   `--load_run`: 指定加载的运行 ID (文件夹名)，默认为 `-1` (最新)。
*   `--env_num`: 训练时的环境数量 (默认 1024)。
*   `--play_env_num`: 回放时的环境数量 (默认 16)。
