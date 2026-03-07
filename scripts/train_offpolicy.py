"""Unified off-policy training entry for SAC and TD3."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from offpolicy_common import build_offpolicy_parser, run_offpolicy


def main() -> None:
    parser = build_offpolicy_parser(
        description="Train off-policy algorithm (SAC/TD3)",
        default_task="Go1JoystickFlatTerrain",
        include_algo=True,
        default_algo="sac",
    )
    args = parser.parse_args()
    run_offpolicy(args.algo, args, ROOT_DIR)


if __name__ == "__main__":
    main()
