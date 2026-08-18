import json
from pathlib import Path
from typing import Any

from .fb.agent import FBAgent, FBModel
from .fb_cpr.agent import FBcprAgent, FBcprModel
from .fb_cpr_aux.agent import FBcprAuxAgent
from .fb_cpr_aux.model import FBcprAuxModel

AGENT_NAME_TO_CLASS = {
    "FBAgent": FBAgent,
    "FBcprAgent": FBcprAgent,
    "FBcprAuxAgent": FBcprAuxAgent,
}
MODEL_NAME_TO_CLASS = {"FBModel": FBModel, "FBcprModel": FBcprModel, "FBcprAuxModel": FBcprAuxModel}


def load_agent_from_checkpoint_dir(checkpoint_dir: str | Path, device: str = "cpu") -> Any:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist.")
    with (checkpoint_path / "config.json").open("r") as f:
        config = json.load(f)

    agent_name = config["name"]
    if agent_name not in AGENT_NAME_TO_CLASS:
        raise ValueError(
            f"Unknown agent name: {agent_name}. Available: {list(AGENT_NAME_TO_CLASS.keys())}"
        )
    agent = AGENT_NAME_TO_CLASS[agent_name].load(str(checkpoint_path), device=device)
    return agent


def load_model_from_checkpoint_dir(checkpoint_dir: str | Path, device: str = "cpu") -> Any:
    checkpoint_path = Path(checkpoint_dir)
    with (checkpoint_path / "model" / "config.json").open("r") as f:
        config = json.load(f)

    model_name = config["name"]
    if model_name not in MODEL_NAME_TO_CLASS:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {list(MODEL_NAME_TO_CLASS.keys())}"
        )
    model = MODEL_NAME_TO_CLASS[model_name].load(str(checkpoint_path / "model"), device=device)
    return model
