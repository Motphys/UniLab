from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.bfm.runner import BFMRunner
from unilab.training import ensure_registries


@hydra.main(version_base="1.3", config_path="../conf/bfm", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    algo_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    train_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.training, resolve=True))
    expert_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.expert, resolve=True))

    runner = BFMRunner(
        algo_cfg=algo_cfg,
        train_cfg=train_cfg,
        expert_cfg=expert_cfg,
    )
    if train_cfg.get("play_only", False):
        runner.play()
    else:
        runner.learn()


if __name__ == "__main__":
    main()
