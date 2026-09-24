# Multi-Node Training

UniLab supports synchronous two-node training with one CUDA GPU per node. The
recipe below is validated on two NVIDIA Spark (GB10) hosts connected by a
200 Gb/s QSFP link; the same procedure applies to any pair of Linux hosts
joined by a private network.

## Prerequisites

- Two hosts with one CUDA GPU each and identical UniLab checkouts and
  environments (`make setup` on both).
- A private network between them. A direct QSFP cable is recommended; a
  switch also works.
- Passwordless SSH from node 0 to node 1 (only needed to start rank 1).

## Connect the two hosts

### 1. Assign static addresses

Connect the cable and pick an unused private /29. On the respective host, run:

```bash
# node 0
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.1/29 ipv6.method disabled
sudo nmcli con up cluster-link

# node 1
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name cluster-link \
  ipv4.method manual ipv4.addresses 192.168.100.2/29 ipv6.method disabled
sudo nmcli con up cluster-link
```

Replace `enp1s0f1np1` with your interface name (`ip link` shows the QSFP
ports). Use a subnet that no other reachable network occupies.

### 2. Verify the link

```bash
ethtool enp1s0f1np1 | grep Speed      # expect 200000Mb/s on QSFP
ping -c3 192.168.100.2                # from node 0
iperf3 -c 192.168.100.2 -t 5 -P 4     # install iperf3 on both nodes first
```

### 3. Set up passwordless SSH

On node 0:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # if you have no key yet
ssh-copy-id <user>@192.168.100.2
```

### 4. NCCL environment

For an Ethernet link, keep NCCL on TCP sockets bound to the cluster interface
on **both** nodes:

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
```

## PPO: two-node torchrun

`algo.num_envs` is a **per-rank** count. For a global budget of `N`
environments, pass `N / 2` on each node. Run the identical command on both
nodes, differing only in `--node_rank`:

```bash
# node 0 (master)
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=192.168.100.1 --master_port=29500 --node_rank=0 \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_flip_tracking/mujoco \
  algo.num_envs=512 algo.max_iterations=1000 \
  training.no_play=true training.log_dir=logs/dual_g1_flip

# node 1: identical command with --node_rank=1
```

Notes:

- Distributed workers require `training.log_dir` to name one shared run
  directory convention; only rank 0 writes TensorBoard, checkpoints, and
  summaries.
- Rank `i` trains with `algo.seed + i`; gradients are averaged after every
  mini-batch, so both ranks hold identical weights throughout.
- Observation-normalizer statistics are rank-local (upstream RSL-RL
  semantics).

## SAC / FlashSAC: external uni_rl data parallelism

Off-policy algorithms use uni_rl's external data-parallel topology instead of
torchrun. Each node runs one complete collector + replay + learner pipeline;
gradients are all-reduced after every backward pass. Pass per-rank values for
`algo.num_envs` and `algo.batch_size` (global divided by two), and export the
topology variables on **both** nodes:

```bash
# node 1
UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE=2 UNILAB_DP_RANK=1 \
UNILAB_DP_RENDEZVOUS_URL=tcp://192.168.100.1:29501 \
UNILAB_DP_LOG_DIR=logs/dual_sac \
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
python src/unilab/scripts/train_sac.py \
  task=g1_walk_flat/mujoco \
  algo.num_envs=1024 algo.batch_size=4096 algo.max_iterations=5000 \
  training.no_play=true training.log_dir=logs/dual_sac

# node 0: identical command with UNILAB_DP_RANK=0
```

Only rank 0 writes checkpoints and logs. The learner keeps both ranks
bitwise-identical through a rank-0 initialization broadcast plus averaged
gradients; each rank replays only transitions collected by its own
environments, which is an unbiased stratification of the single-node replay.

## What to expect on two Spark (GB10) hosts

| Workload | Dual-node speedup |
| --- | --- |
| PPO g1_flip_tracking 1024 envs | 1.34× |
| PPO go2_joystick_flat 2048 envs | 1.44× |
| PPO g1_walk_flat 2048 envs | 1.55× |
| PPO allegro_inhand 16384 envs | 1.98× |
| FlashSAC g1_motion_tracking, global batch 8192 | 1.75× |

Off-policy speedup depends on the compute/communication ratio: small-MLP
learners at small batch may not speed up (single-GPU learner is not yet
saturated), while larger batches and heavier learners scale. See the
[dual-Spark validation report](https://github.com/Motphys/UniLab/blob/feat/dual-spark/docs/reports/dual_spark_2026-09.md)
for full measurements and the batch-size crossover analysis.
