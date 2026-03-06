"""Batch coordination for collector-trainer synchronization."""

import multiprocessing as mp

_SPAWN_CTX = mp.get_context("spawn")


class BatchCoordinator:
    """Coordinates collector and trainer to work in batches."""

    def __init__(self, create: bool = True):
        if create:
            self._collector_sem = _SPAWN_CTX.Semaphore(1)
            self._trainer_sem = _SPAWN_CTX.Semaphore(0)
