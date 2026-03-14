from typing import Any, Dict, Tuple


class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, Tuple[Any, Any, Any]] = {}

    def register(self, name: str, env_cls, env_cfg, train_cfg):
        self._tasks[name] = (env_cls, env_cfg, train_cfg)

    def get_task(self, name: str):
        if name not in self._tasks:
            raise ValueError(
                f"Task '{name}' not found. Available tasks: {list(self._tasks.keys())}"
            )
        return self._tasks[name]


task_registry = TaskRegistry()
