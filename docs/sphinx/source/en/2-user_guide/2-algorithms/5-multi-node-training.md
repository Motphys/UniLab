# Multi-Node Training

UniLab supports synchronous multi-node training on clusters of single-GPU
Linux hosts. This guide walks through the complete setup from connecting two
NVIDIA Spark (GB10) hosts with a QSFP cable to launching training and
verifying results. The same procedure applies to any pair of Linux hosts
joined by a private network.

```{contents} Contents
:local:
```

## Step 1: Connect the two hosts

### 1.1 Physically connect

Plug a QSFP cable between the two hosts. On NVIDIA Spark the QSFP ports
appear as `enp1s0f0np0`, `enp1s0f1np1`, `enP2p1s0f0np0`, and
`enP2p1s0f1np1`. Use the same port number on both hosts (the right-side
ports `enp1s0f1np1` are recommended). Check the link is up:

```bash
ip link show enp1s0f1np1    # should show "UP" and "LOWER_UP"
ethtool enp1s0f1np1 | grep -E 'Speed|Link'   # expect "200000Mb/s" and "yes"
```

### 1.2 Assign static IP addresses

Pick an unused private /29 subnet (e.g. `192.168.100.0/29`). On the
respective host:

```bash
# Host 0 (spark0 — the master)
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.1/29 ipv6.method disabled
sudo nmcli con up cluster-link

# Host 1 (spark1 — the worker)
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.2/29 ipv6.method disabled
sudo nmcli con up cluster-link
```

> Replace `enp1s0f1np1` with your actual interface name (`ip link` lists all
> ports). Use a subnet that no other reachable network occupies.

### 1.3 Verify connectivity

```bash
# From spark0
ping -c3 192.168.100.2                     # should reply
sudo apt install -y iperf3                 # on both hosts
iperf3 -c 192.168.100.2 -t 5 -P 4          # expect >50 Gbit/s on QSFP
```

### 1.4 Set up passwordless SSH

On spark0:

```bash
# Generate a key if you don't have one
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# Copy the key to spark1 (enter spark1's password once)
ssh-copy-id <user>@192.168.100.2

# Add an SSH alias so you can just type "ssh spark1"
cat >> ~/.ssh/config <<'EOF'
Host spark1
  HostName 192.168.100.2
  User <user>
EOF

# Verify: should return spark1's hostname without a password prompt
ssh spark1 hostname
```

## Step 2: Clone and set up both repositories

Run these commands **on both hosts** (paths must be identical):

```bash
mkdir -p ~/ws/motphys && cd ~/ws/motphys

git clone -b feat/dual-spark https://github.com/Motphys/UniLab.git
git clone -b feat/dual-spark https://github.com/unilabsim/unilab_rl.git

cd UniLab

# Install dependencies (torch, mujoco, uni_rl, etc.)
make setup
# Or equivalently:
# uv sync --extra mujoco --extra uni_rl
# uv run --no-sync unilab-complete install

# Point uni_rl at the local patched checkout
uv pip install --no-deps -e ../unilab_rl

# Verify
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"  # True
uv run --no-sync python -c "from uni_rl.ipc.dp_launcher import current_external_dp_topology; print('ok')"
```

> **Important**: after any `uv sync` or `make setup`, re-run
> `uv pip install --no-deps -e ../unilab_rl` — the sync restores the PyPI
> version of unilab-rl which lacks the multi-node patches.

## Step 3: Launch training

The repository provides `scripts/launch_distributed.py` — a single command
on spark0 that starts rank 0 locally and SSHes into spark1 to start rank 1.

### 3.1 PPO

```bash
cd ~/ws/motphys/UniLab

# Single node (no cluster arguments needed)
uv run --no-sync python scripts/launch_distributed.py \
  --algo ppo --task g1_walk_flat --sim mujoco

# Two nodes
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
# Two nodes, FlashSAC with full single-machine scale per node
uv run --no-sync python scripts/launch_distributed.py \
  --algo flashsac --task g1_walk_flat --sim mujoco \
  --ifname enp1s0f1np1 \
  --master-ip 192.168.100.1 \
  --peer spark1 \
  --num-nodes 2 \
  algo.num_envs=4096 algo.batch_size=8192
```

### 3.3 Command-line arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--algo` | always | `ppo`, `sac`, or `flashsac` |
| `--task` | always | Task name (e.g. `g1_walk_flat`) |
| `--sim` | optional | Backend (default: `mujoco`) |
| `--ifname` | multi-node | QSFP interface name (must be the same on both hosts) |
| `--master-ip` | multi-node | spark0's IP on the QSFP link |
| `--peer` | multi-node | SSH alias/IP of spark1 (repeat for >2 nodes) |
| `--num-nodes` | optional | Number of hosts (default 1 = single node) |
| `--remote-dir` | optional | UniLab path on peers (default `~/ws/motphys/UniLab`) |
| `--port` | optional | torchrun rendezvous port (default 29500) |
| `--dp-port` | optional | off-policy TCPStore port (default 29501) |
| overrides | optional | Hydra overrides (e.g. `algo.num_envs=2048`) |

Run `python scripts/launch_distributed.py --help` for the full reference.

## Step 4: Verify training is running

After launching, you should see:

```text
[launch] algo=ppo task=g1_walk_flat/mujoco nodes=2 log_dir=logs/distributed_ppo_...
[launch] rank 1 -> spark1
[launch] rank 0 local
```

Check both GPUs are active (from spark0):

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
ssh spark1 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

TensorBoard logs are written by rank 0 only:

```bash
uv run --no-sync tensorboard --logdir logs --port 6006
```

Check the run summary after training completes:

```bash
cat logs/<run-dir>/run_summary.json | python -m json.tool | grep -E 'status|world_size|total_env_steps'
# Expect: "status": "completed", "world_size": 2
```

## Expected throughput on two Spark (GB10) hosts

When each node runs at the **full single-machine scale** (same `num_envs`
and `batch_size` as the single-node baseline), system throughput scales:

| Workload | Per-node scale | Throughput gain |
| --- | --- | --- |
| PPO g1_flip_tracking | 1024 envs | 1.78× |
| PPO go2_joystick_flat | 2048 envs | 1.93× |
| PPO g1_walk_flat | 2048 envs | 1.87× |
| PPO allegro_inhand | 16384 envs | 1.97× |
| FlashSAC g1_walk_flat | 4096 envs, batch 8192 | 1.91× |
| FlashSAC g1_motion_tracking | 2048 envs, batch 8192 | 1.90× |
| SAC g1_walk_flat | 2048 envs, batch 8192 | 1.42× |

Throughput approaches 2× as the learner compute weight grows (FlashSAC,
large-batch, large networks) or as the rollout load grows (more envs).
Lightweight learners (small MLP + small batch) are limited to ~1.4× by the
fixed cost of gradient synchronization. See the
[dual-Spark throughput report](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md)
for full measurements.

## Manual launch (without the script)

If you prefer to run commands on each node manually:

### PPO (torchrun, identical on both nodes except `--node_rank`)

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

# spark1: identical with --node_rank=1
```

### SAC / FlashSAC (external uni_rl DP, identical except `UNILAB_DP_RANK`)

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

# spark0: identical with UNILAB_DP_RANK=0
```
