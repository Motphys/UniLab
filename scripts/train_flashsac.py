"""FlashSAC training/playback entrypoint (shared implementation in train_offpolicy.py)."""

import hydra
from omegaconf import DictConfig
from train_offpolicy import main as _offpolicy_main


@hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config")
def main(cfg: DictConfig) -> None:
    _offpolicy_main(cfg)


if __name__ == "__main__":
    main()
