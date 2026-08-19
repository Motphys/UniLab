# BFM

BFM (Body-Frame Motion) 通过未来预测辅助 critic (FB-CPR-Aux) 从动作捕捉数据学习人形机器人运动。
训练脚本为 `scripts/train_bfm.py`，主配置位于 `conf/bfm/config.yaml`，
算法默认值在同一目录下。当前日志名称为 `bfm`。

## 运行模型

BFM 使用 off-policy 学习与优先级轨迹回放。trajectory buffer 存储完整 episode
(seq_length=8) 及 body-frame 观测。专家动作数据 (LAFAN1) 在首次运行时从 HuggingFace 下载。
agent 架构包含用于动作模仿的判别器、未来预测辅助 critic 和带 body-frame 编码的 actor。

## 快速开始

```bash
# CUDA (MuJoCo 后端)
uv run train --algo bfm --task g1_bfm --sim mujoco

# CUDA (Motrix 后端)
uv run train --algo bfm --task g1_bfm --sim motrix

# ROCm
make sync-rocm
HIP_VISIBLE_DEVICES=0 uv run train --algo bfm --task g1_bfm --sim mujoco
```

## 关键字段

- `algo.algo_log_name=bfm`
- `algo.num_envs=1024`
- `algo.num_env_steps=1536000000` (总计 1.536B 步)
- `algo.buffer_size=5120000` (5.12M transitions)
- `algo.buffer_device=cuda` (或 `cpu` 以节省 GPU 显存)
- `algo.use_trajectory_buffer=true` (基于轨迹的 replay)
- `algo.model.seq_length=8` (轨迹序列长度)

覆盖参数：

```bash
uv run train --algo bfm --task g1_bfm --sim mujoco \
  algo.num_envs=2048 \
  algo.buffer_size=10000000 \
  algo.num_env_steps=500000000
```

## 后端

- **MuJoCo**: CPU 物理，<2048 envs 时更快
- **Motrix**: GPU 加速物理，可扩展至 4096+ envs

## 故障排查

### ROCm: "Found no NVIDIA driver"

安装了 CUDA torch 而非 ROCm。修复：

```bash
make sync-rocm
```

### "libmujoco.so.3.X.Y: cannot open shared object file"

原生扩展针对错误的 mujoco 版本编译。重新构建：

```bash
uv pip install --python .venv/bin/python --no-build-isolation \
    --no-binary mujoco-uni-runtime --force-reinstall --no-cache \
    mujoco-uni-runtime==0.3.1
```

### "No trajectories with length >= 9"

`num_seed_steps` 对 trajectory buffer 来说太小。增加：

```bash
uv run train --algo bfm --task g1_bfm --sim mujoco algo.num_seed_steps=4096
```
