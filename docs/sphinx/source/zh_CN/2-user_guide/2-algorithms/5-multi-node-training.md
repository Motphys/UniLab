# 多节点训练

UniLab 支持每节点一块 CUDA GPU 的双机同步训练。下面的流程在两台通过
200 Gb/s QSFP 直连的 NVIDIA Spark（GB10）上完成验证；同样适用于任何由
私有网络连接的两台 Linux 主机。

## 前提条件

- 两台各有一块 CUDA GPU 的主机，UniLab 检出与环境完全一致（两边都执行
  `make setup`）。
- 两机之间的私有网络：推荐 QSFP 直连，经交换机亦可。
- 从节点 0 到节点 1 的 SSH 免密（仅用于启动 rank 1）。

## 连接两台主机

### 1. 配置静态地址

插好线缆并选择一个未被占用的私有 /29 网段，在对应主机上执行：

```bash
# 节点 0
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.1/29 ipv6.method disabled
sudo nmcli con up cluster-link

# 节点 1
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.2/29 ipv6.method disabled
sudo nmcli con up cluster-link
```

将 `enp1s0f1np1` 替换为你的接口名（`ip link` 可查看 QSFP 端口）。网段不能与
任何现有可达网络重叠。

### 2. 验证链路

```bash
ethtool enp1s0f1np1 | grep Speed      # QSFP 期望 200000Mb/s
ping -c3 192.168.100.2                # 在节点 0 上执行
iperf3 -c 192.168.100.2 -t 5 -P 4     # 两边先安装 iperf3
```

### 3. 配置 SSH 免密

在节点 0 上：

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # 若尚无密钥
ssh-copy-id <user>@192.168.100.2
```

### 4. NCCL 环境

以太网链路下，让 NCCL 使用绑定集群网卡的 TCP socket，在**两台**节点上：

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
```

## PPO：双机 torchrun

`algo.num_envs` 是**每 rank** 数量。若目标是全局 `N` 个环境，每台节点传
`N / 2`。两台执行完全相同的命令，仅 `--node_rank` 不同：

```bash
# 节点 0（master）
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=192.168.100.1 --master_port=29500 --node_rank=0 \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_flip_tracking/mujoco \
  algo.num_envs=512 algo.max_iterations=1000 \
  training.no_play=true training.log_dir=logs/dual_g1_flip

# 节点 1：同命令，--node_rank=1
```

说明：

- 分布式 worker 要求 `training.log_dir` 指定一个统一的 run 目录约定；只有
  rank 0 写 TensorBoard、checkpoint 与 summary。
- rank `i` 使用 `algo.seed + i` 训练；每个 mini-batch 后梯度取平均，因此
  两 rank 的权重始终一致。
- 观测归一化统计是 rank-local 的（RSL-RL 上游语义）。

## SAC / FlashSAC：uni_rl 外部数据并行

off-policy 算法使用 uni_rl 的外部数据并行拓扑而非 torchrun。每台节点运行
一套完整的 collector + replay + learner 流水线，并在每次 backward 后
all-reduce 梯度。`algo.num_envs` 与 `algo.batch_size` 传每 rank 值（全局量
除以二），并在**两台**节点上导出拓扑变量：

```bash
# 节点 1
UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK=1 \
UNILAB_DP_RENDEZVOUS_URL=tcp://192.168.100.1:29501 \
UNILAB_DP_LOG_DIR=logs/dual_sac \
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco \
  algo.num_envs=1024 algo.batch_size=4096 algo.max_iterations=5000 \
  training.no_play=true training.log_dir=logs/dual_sac

# 节点 0：同命令，UNILAB_DP_RANK=0
```

只有 rank 0 写 checkpoint 与日志。learner 通过 rank 0 初始化广播加梯度平均
保持两 rank 逐位一致；每个 rank 只回放自己环境采集的 transition，是单机
replay 的无偏分层抽样。

## 两台 Spark（GB10）上的预期表现

| 负载 | 双机加速 |
| --- | --- |
| PPO g1_flip_tracking 1024 envs | 1.34× |
| PPO go2_joystick_flat 2048 envs | 1.44× |
| PPO g1_walk_flat 2048 envs | 1.55× |
| PPO allegro_inhand 16384 envs | 1.98× |
| FlashSAC g1_motion_tracking，全局 batch 8192 | 1.75× |

off-policy 的加速取决于计算/通信比：小 MLP learner 在小 batch 下可能无法
加速（单卡 learner 尚未饱和），更大的 batch 与更重的 learner 才有扩展性。
完整测量与 batch 交叉点分析见
[双 Spark 验证报告](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md)。
