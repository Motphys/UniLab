import typing as tp

import gymnasium.spaces as gym_spaces
import numpy as np

SUPPORTED_SPACES = tp.Union[gym_spaces.Box, gym_spaces.Discrete, gym_spaces.Dict]


def space_to_json(space: SUPPORTED_SPACES):

    if isinstance(space, gym_spaces.Box):
        state = {
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype),
            "type": "Box",
        }
        return state
    elif isinstance(space, gym_spaces.Discrete):
        state = {"n": space.n, "type": "Discrete"}
        if space.start != 0:
            state["start"] = int(space.start)
        return state
    elif isinstance(space, gym_spaces.Dict):
        state = dict(type="Dict", spaces=dict())
        for name, sub_space in space.spaces.items():
            space_json = space_to_json(sub_space)
            state["spaces"][name] = space_json
        return state
    else:
        raise NotImplementedError(
            f"Space type {type(space)} is not supported for JSON serialization."
        )


def json_to_space(json_data: tp.Dict[str, tp.Any]) -> SUPPORTED_SPACES:

    if json_data["type"] == "Box":
        dtype = np.dtype(json_data["dtype"])
        return gym_spaces.Box(
            low=np.array(json_data["low"], dtype=dtype),
            high=np.array(json_data["high"], dtype=dtype),
            shape=json_data.get("shape", None),
            dtype=dtype,
        )
    if json_data["type"] == "Discrete":
        if "start" in json_data:
            return gym_spaces.Discrete(n=json_data["n"], start=json_data["start"])
        else:
            return gym_spaces.Discrete(n=json_data["n"])
    if json_data["type"] == "Dict":
        spaces = {}
        for name, space_json in json_data["spaces"].items():
            spaces[name] = json_to_space(space_json)
        dict_space = gym_spaces.Dict(spaces=spaces)
        return dict_space
    else:
        raise NotImplementedError(
            f"Space type {json_data['type']} is not supported for JSON deserialization."
        )
