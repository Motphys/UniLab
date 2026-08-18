import numpy as np
import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten, tree_map


def clone_if_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


def tree_clone(pytree):

    return tree_map(clone_if_tensor, pytree)


def tree_check_batch_size(pytree, batch_size, prefix=""):

    if isinstance(pytree, (list, tuple)):
        for i, item in enumerate(pytree):
            tree_check_batch_size(item, batch_size, prefix=f"{prefix}[{i}]")
    elif isinstance(pytree, dict):
        for key, item in pytree.items():
            tree_check_batch_size(item, batch_size, prefix=f"{prefix}.{key}")
    elif isinstance(pytree, torch.Tensor):
        if pytree.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch at {prefix}: expected {batch_size}, got {pytree.shape[0]}"
            )


def tree_get_batch_size(pytree):
    tensors, _ = tree_flatten(pytree)
    batch_sizes = [t.shape[0] for t in tensors]
    assert all(bs == batch_sizes[0] for bs in batch_sizes), (
        f"All tensors must have the same batch size {batch_sizes[0]}, got {batch_sizes}"
    )
    return batch_sizes[0]


def tree_numpy_to_tensor(pytree):

    def convert(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    return tree_map(convert, pytree)


def tree_concat(list_of_pytree_of_tensors, dim=0):

    tds = tuple(
        map(lambda x: TensorDict.from_pytree(x, auto_batch_size=True), list_of_pytree_of_tensors)
    )
    concatenated = torch.cat(tds, dim=dim)

    if isinstance(concatenated, TensorDict):
        return concatenated.to_pytree()

    return concatenated


def tree_concat_numpy(list_of_pytree_of_arrays, dim=0):
    concatenated = tree_concat(list_of_pytree_of_arrays, dim=dim)
    return tree_map(lambda x: x.numpy() if isinstance(x, torch.Tensor) else x, concatenated)
