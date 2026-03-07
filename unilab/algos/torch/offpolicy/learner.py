"""Base interface for off-policy RL algorithms."""

from abc import ABC, abstractmethod
from typing import Dict
import torch


class OffPolicyLearner(ABC):
    """Unified interface for off-policy RL learners."""

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        pass

    @abstractmethod
    def get_actor_state_dict(self) -> Dict:
        """Get actor weights for syncing to collector."""
        pass

    @abstractmethod
    def load_actor_state_dict(self, state_dict: Dict):
        """Load actor weights from collector sync."""
        pass
