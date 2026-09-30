#!/usr/bin/env python3
"""Launch UniLab training on one or more Spark hosts.

Single node:
    python scripts/launch_distributed.py --algo ppo --task g1_walk_flat

Two nodes:
    python scripts/launch_distributed.py --algo ppo --task g1_walk_flat \\
        --ifname enp1s0f1np1 --master-ip 192.168.100.1 \\
        --peer spark2 --num-nodes 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Sequence


_ENTRYPOINT = {
    "ppo": "src/unilab/scripts/train_rsl_rl.py",
    "sac": "src/unilab/scripts/train_sac.py",
    "flashsac": "src/unilab/scripts/train_flashsac.py",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "flashsac"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--sim", default="mujoco")
    parser.add_argument("overrides", nargs="*", default=[],
                        help="Hydra overrides (e.g. algo.num_envs=2048)")

    cluster = parser.add_argument_group("cluster (required when --num-nodes > 1)")
    cluster.add_argument("-N", "--num-nodes", type=int, default=1)
    cluster.add_argument("--ifname", help="QSFP interface name")
    cluster.add_argument("--master-ip", help="spark0 (rank 0) IP on the QSFP link")
    cluster.add_argument("--peer", action="append", default=[],
                         help="SSH alias/IP of each peer (repeat for >2 nodes)")
    cluster.add_argument("--port", type=int, default=29500,
                         help="torchrun rendezvous port (default: 29500)")
    cluster.add_argument("--dp-port", type=int, default=29501,
                         help="off-policy TCPStore port (default: 29501)")
    cluster.add_argument("--log-dir", default=None)
    cluster.add_argument("--remote-dir", default="~/ws/motphys/UniLab",
                         help="UniLab checkout path on peer hosts (default: ~/ws/motphys/UniLab)")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.num_nodes < 1:
        raise SystemExit("error: --num-nodes must be >= 1")
    if args.num_nodes == 1:
        return
    missing = []
    if not args.ifname:
        missing.append("--ifname")
    if not args.master_ip:
        missing.append("--master-ip")
    if len(args.peer) != args.num_nodes - 1:
        missing.append(f"--peer (expected {args.num_nodes - 1}, got {len(args.peer)})")
    if missing:
        raise SystemExit(
            f"error: multi-node mode requires: {', '.join(missing)}\n"
            "       see --help for details"
        )


def _log_dir(args: argparse.Namespace) -> str:
    if args.log_dir:
        return args.log_dir
    return f"logs/distributed_{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}"


def _hydra_args(args: argparse.Namespace, log_dir: str) -> list[str]:
    return [
        f"task={args.task}/{args.sim}",
        f"training.log_dir={log_dir}",
        "training.no_play=true",
        *args.overrides,
    ]


def _launch_single(args: argparse.Namespace, log_dir: str) -> int:
    cmd = [sys.executable, _ENTRYPOINT[args.algo], *_hydra_args(args, log_dir)]
    print(f"[launch] single-node: {' '.join(cmd[:3])}...")
    return subprocess.run(cmd, check=False).returncode


def _peer_ssh_command(args: argparse.Namespace, rank: int, log_dir: str) -> str:
    overrides = " ".join(f"'{o}'" for o in _hydra_args(args, log_dir))
    nccl = f"NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME={args.ifname}"
    cd = f"cd {args.remote_dir} &&"
    py = ".venv/bin/python"

    if args.algo == "ppo":
        return (
            f"{cd} {nccl} {py} -m torch.distributed.run"
            f" --nnodes={args.num_nodes} --nproc_per_node=1"
            f" --master_addr={args.master_ip} --master_port={args.port}"
            f" --node_rank={rank}"
            f" {_ENTRYPOINT['ppo']} {overrides}"
        )
    return (
        f"{cd} {nccl}"
        f" UNILAB_DP_EXTERNAL=1 UNILAB_DP_WORLD_SIZE={args.num_nodes}"
        f" UNILAB_DP_RANK={rank}"
        f" UNILAB_DP_RENDEZVOUS_URL=tcp://{args.master_ip}:{args.dp_port}"
        f" UNILAB_DP_LOG_DIR={log_dir}"
        f" {py} {_ENTRYPOINT[args.algo]} {overrides}"
    )


def _launch_multi(args: argparse.Namespace, log_dir: str) -> int:
    peers: list[subprocess.Popen] = []
    for rank, peer in enumerate(args.peer, start=1):
        ssh_cmd = _peer_ssh_command(args, rank, log_dir)
        print(f"[launch] rank {rank} -> {peer}")
        peers.append(subprocess.Popen(
            ["ssh", "-tt", peer, ssh_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        ))

    env = dict(os.environ, NCCL_IB_DISABLE="1", NCCL_SOCKET_IFNAME=args.ifname)
    if args.algo == "ppo":
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", str(args.num_nodes),
            "--nproc_per_node", "1",
            "--master_addr", args.master_ip,
            "--master_port", str(args.port),
            "--node_rank", "0",
            _ENTRYPOINT["ppo"],
            *_hydra_args(args, log_dir),
        ]
    else:
        env.update({
            "UNILAB_DP_EXTERNAL": "1",
            "UNILAB_DP_WORLD_SIZE": str(args.num_nodes),
            "UNILAB_DP_RANK": "0",
            "UNILAB_DP_RENDEZVOUS_URL": f"tcp://{args.master_ip}:{args.dp_port}",
            "UNILAB_DP_LOG_DIR": log_dir,
        })
        cmd = [sys.executable, _ENTRYPOINT[args.algo], *_hydra_args(args, log_dir)]

    print(f"[launch] rank 0 local")
    try:
        return subprocess.run(cmd, env=env, check=False).returncode
    except KeyboardInterrupt:
        print("\n[launch] interrupted; stopping peers...")
        return 130
    finally:
        for proc in peers:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate(args)
    log_dir = _log_dir(args)
    nodes = args.num_nodes
    print(f"[launch] algo={args.algo} task={args.task}/{args.sim} nodes={nodes} log_dir={log_dir}")
    if nodes == 1:
        return _launch_single(args, log_dir)
    return _launch_multi(args, log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
