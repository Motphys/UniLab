import json
import pickle
import typing as tp
from pathlib import Path

import torch
from torch import nn

from .base import BaseConfig
from .envs.utils.gym_spaces import json_to_space, space_to_json


def save_model(
    path: str, model: "BaseModel", build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
) -> None:
    import safetensors.torch

    output_folder = Path(path)
    output_folder.mkdir(exist_ok=True)
    safetensors.torch.save_model(model, output_folder / "model.safetensors")

    json_dump = model.cfg.model_dump()

    if build_kwargs is not None:
        if "obs_space" in build_kwargs:
            build_kwargs["obs_space"] = space_to_json(build_kwargs["obs_space"])
        with (output_folder / "init_kwargs.json").open("w+") as f:
            json.dump(build_kwargs, f, indent=4)

    with (output_folder / "config.json").open("w+") as f:
        f.write(json.dumps(json_dump, indent=4))


def load_model(
    path: str,
    device: str | None,
    strict: bool,
    config_class: "BaseModelConfig",
    build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> "BaseModel":
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device

    if (model_dir / "init_kwargs.pkl").exists():
        with (model_dir / "init_kwargs.pkl").open("rb") as f:
            build_kwargs = pickle.load(f)
    elif (model_dir / "init_kwargs.json").exists():
        with (model_dir / "init_kwargs.json").open("r") as f:
            build_kwargs = json.load(f)
            if "obs_space" in build_kwargs:
                build_kwargs["obs_space"] = json_to_space(build_kwargs["obs_space"])

    if build_kwargs is None:
        raise ValueError(
            "No build_kwargs provided, and init_kwargs.pkl not found. Please provide build_kwargs that are passed to config_class.build functionm."
        )

    loaded_config = config_class(**loaded_config)
    loaded_model = loaded_config.build(**build_kwargs)

    import safetensors.torch

    state_dict = safetensors.torch.load_file(model_dir / "model.safetensors", device=device)
    if strict and any(["target" in key for key in state_dict.keys()]):
        loaded_model._prepare_for_train()
    strict = False
    loaded_model.load_state_dict(state_dict, strict=strict)
    return loaded_model


class BaseModelConfig(BaseConfig):
    device: tp.Literal["cpu", "cuda"] = "cuda"


class BaseModel(nn.Module):
    config_class = BaseModelConfig

    def __init__(self, obs_space, action_dim, config: BaseModelConfig):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = config

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device.type
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True):
        return load_model(path, device, strict=strict, config_class=cls.config_class)

    def save(self, output_folder: str) -> None:
        return save_model(
            output_folder,
            self,
            build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim},
        )
