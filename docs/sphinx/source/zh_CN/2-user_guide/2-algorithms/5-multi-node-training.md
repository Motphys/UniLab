# 多节点训练

UniLab 支持在单 GPU Linux 主机集群上进行同步多节点训练。本指南从两台
NVIDIA Spark（GB10）通过 QSFP 线缆连接，到克隆仓库、配置环境、启动训练
和验证结果的完整流程。同样适用于任何由私有网络连接的两台 Linux 主机。

```{contents} 目录
:local:
```

## 第一步：连接两台主机

### 1.1 物理连接

用 QSFP 线缆连接两台主机。NVIDIA Spark 的 QSFP 端口在系统中显示为
`enp1s0f0np0`、`enp1s0f1np1`、`enP2p1s0f0np0` 和 `enP2p1s0f1np1`。
建议两台使用相同编号的端口（推荐右侧的 `enp1s0f1np1`）。检查链路：

```bash
ip link show enp1s0f1np1    # 应显示 "UP" 和 "LOWER_UP"
ethtool enp1s0f1np1 | grep -E 'Speed|Link'   # 期望 "200000Mb/s" 和 "yes"
```

### 1.2 配置静态 IP

选择一个未被占用的私有 /29 网段（如 `192.168.100.0/29`），在对应主机上执行：

```bash
# 主机 0（spark0 — 主节点）
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.1/29 ipv6.method disabled
sudo nmcli con up cluster-link

# 主机 1（spark1 — 工作节点）
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.2/29 ipv6.method disabled
sudo nmcli con up cluster-link
```

> 将 `enp1s0f1np1` 替换为实际接口名（`ip link` 查看全部端口）。
> 网段不能与任何现有可达网络重叠。

### 1.3 验证连通性

```bash
# 在 spark0 上
ping -c3 192.168.100.2                     # 应能 ping 通
sudo apt install -y iperf3                 # 两台都装
iperf3 -c 192.168.100.2 -t 5 -P 4          # QSFP 期望 >50 Gbit/s
```

### 1.4 配置 SSH 免密

在 spark0 上：

```bash
# 如尚无密钥则生成
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# 将公钥复制到 spark1（输入一次 spark1 的密码）
ssh-copy-id <user>@192.168.100.2

# 添加 SSH 别名
cat >> ~/.ssh/config <<'EOF'
Host spark1
  HostName 192.168.100.2
  User <user>
EOF

# 验证：应直接返回 spark1 的主机名，不提示密码
ssh spark1 hostname
```

## 第二步：克隆并配置两个仓库

在**两台主机上**执行以下命令（路径必须一致）：

```bash
mkdir -p ~/ws/motphys && cd ~/ws/motphys

git clone -b feat/dual-spark https://github.com/Motphys/UniLab.git
git clone -b feat/dual-spark https://github.com/unilabsim/unilab_rl.git

cd UniLab

# 安装依赖（torch、mujoco、uni_rl 等）
make setup

# 将 uni_rl 指向本地补丁版
uv pip install --no-deps -e ../unilab_rl

# 验证
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"  # True
uv run --no-sync python -c "from uni_rl.ipc.dp_launcher import current_external_dp_topology; print('ok')"
```

> **注意**：每次执行 `uv sync` 或 `make setup` 后，必须重新执行
> `uv pip install --no-deps -e ../unilab_rl`——sync 会恢复 PyPI 版本的
> unilab-rl，该版本不包含多节点补丁。

## 第三步：启动训练

仓库提供 `scripts/launch_distributed.py` 启动脚本——只需在 spark0 上
执行一条命令，脚本会本机启动 rank 0 并通过 SSH 在 spark1 上启动 rank 1。

### 3.1 PPO

```bash
cd ~/ws/motphys/UniLab

# 单机（无需任何集群参数）
uv run --no-sync python scripts/launch_distributed.py \
  --algo ppo --task g1_walk_flat --sim mujoco

# 双机
uv run --no-sync python scripts/launch_distributed.py \
  --algo ppo --task g1_walk_flat --sim mujoco \
  --ifname enp1s0f1np1 \
  --master-ip 192.168.100.1 \
  --peer spark1 \
  --num-nodes 2 \
  algo.num_envs=2048 algo.max_iterations=2200
```

### 3.2 SAC / FlashSAC

```bash
# 双机 FlashSAC（每节点按单机同等规模满配）
uv run --no-sync python scripts/launch_distributed.py \
  --algo flashsac --task g1_walk_flat --sim mujoco \
  --ifname enp1s0f1np1 \
  --master-ip 192.168.100.1 \
  --peer spark1 \
  --num-nodes 2 \
  algo.num_envs=4096 algo.batch_size=8192
```

### 3.3 命令行参数

| 参数 | 必传 | 说明 |
| --- | --- | --- |
| `--algo` | 始终 | `ppo`、`sac` 或 `flashsac` |
| `--task` | 始终 | 任务名（如 `g1_walk_flat`） |
| `--sim` | 可选 | 仿真后端（默认 `mujoco`） |
| `--ifname` | 多节点 | QSFP 网卡名（两台必须一致） |
| `--master-ip` | 多节点 | spark0 在 QSFP 网段的 IP |
| `--peer` | 多节点 | spark1 的 SSH 别名/IP（多节点可重复传） |
| `--num-nodes` | 可选 | 节点数（默认 1 = 单机） |
| `--remote-dir` | 可选 | 远端 UniLab 路径（默认 `~/ws/motphys/UniLab`） |
| `--port` | 可选 | torchrun 端口（默认 29500） |
| `--dp-port` | 可选 | off-policy TCPStore 端口（默认 29501） |
| overrides | 可选 | Hydra 参数透传（如 `algo.num_envs=2048`） |

运行 `python scripts/launch_distributed.py --help` 查看完整帮助。

## 第四步：验证训练正在运行

启动后应看到：

```text
[launch] algo=ppo task=g1_walk_flat/mujoco nodes=2 log_dir=logs/distributed_ppo_...
[launch] rank 1 -> spark1
[launch] rank 0 local
```

检查两台 GPU 都在工作（在 spark0 上）：

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
ssh spark1 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

TensorBoard 日志只由 rank 0 写入：

```bash
uv run --no-sync tensorboard --logdir logs --port 6006
```

训练完成后检查 run summary：

```bash
cat logs/<run-dir>/run_summary.json | python -m json.tool | grep -E 'status|world_size|total_env_steps'
# 期望："status": "completed", "world_size": 2
```

## 两台 Spark（GB10）上的预期吞吐

每节点按**单机同等规模满配**（`num_envs` 和 `batch_size` 与单机基线相同）时，系统吞吐：

| 负载 | 每节点规模 | 吞吐提升 |
| --- | --- | --- |
| PPO g1_flip_tracking | 1024 envs | 1.78× |
| PPO go2_joystick_flat | 2048 envs | 1.93× |
| PPO g1_walk_flat | 2048 envs | 1.87× |
| PPO allegro_inhand | 16384 envs | 1.97× |
| FlashSAC g1_walk_flat | 4096 envs, batch 8192 | 1.91× |
| FlashSAC g1_motion_tracking | 2048 envs, batch 8192 | 1.90× |
| SAC g1_walk_flat | 2048 envs, batch 8192 | 1.42× |

吞吐随 learner 计算重量（FlashSAC、大 batch、大网络）或 rollout 负载
（更多 env）的增长而接近 2×。轻量 learner（小 MLP + 小 batch）受梯度
同步固定开销限制约 1.4×。详见
[双 Spark 吞吐报告](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md)。

## 手动启动（不使用脚本）

如需在各节点手动执行命令：

### PPO（torchrun，两台命令相同仅 `--node_rank` 不同）

```bash
# spark0
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=192.168.100.1 --master_port=29500 --node_rank=0 \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_flip_tracking/mujoco \
  algo.num_envs=512 algo.max_iterations=1000 \
  training.no_play=true training.log_dir=logs/dual_g1_flip

# spark1：同命令，--node_rank=1
```

### SAC / FlashSAC（外部 DP，两台命令相同仅 `UNILAB_DP_RANK` 不同）

```bash
# spark1
UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK=1 \
UNILAB_DP_RENDEZVOUS_URL=tcp://192.168.100.1:29501 \
UNILAB_DP_LOG_DIR=logs/dual_sac \
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco \
  algo.num_envs=1024 algo.batch_size=4096 \
  training.no_play=true training.log_dir=logs/dual_sac

# spark0：同命令，UNILAB_DP_RANK=0
```
