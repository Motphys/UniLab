from typing import Any, Literal

import pydantic


class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="forbid", strict=True, use_enum_values=True, frozen=True
    )

    name: Literal["BaseConfig"] = "BaseConfig"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name") or cls.name == "BaseConfig":
            cls.name = cls.__name__
            cls.__annotations__["name"] = Literal[cls.__name__]
        else:
            cls.__annotations__["name"] = Literal[
                (cls.__name__,) + cls.__annotations__["name"].__args__
            ]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in config {self.__class__.__name__}")

    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"The object {self} did not have valid build function. Did you forget to define it?"
        )
