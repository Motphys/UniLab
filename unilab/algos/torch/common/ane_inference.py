"""Simplified ANE backend using deterministic inference."""

import numpy as np


def create_ane_actor(actor_model, obs_dim, action_dim):
    """Create ANE-compatible actor using deterministic inference.

    Uses mean action (no sampling) to avoid CoreML limitations.
    """
    try:
        import coremltools as ct
        import torch

        # Create deterministic wrapper
        class DeterministicActor(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                # Get mean action (deterministic)
                with torch.no_grad():
                    mean, _ = self.actor.forward(obs)
                    return torch.tanh(mean)

        det_actor = DeterministicActor(actor_model)
        det_actor.eval()

        # Trace model
        example = torch.randn(1, obs_dim)
        traced = torch.jit.trace(det_actor, example)

        # Convert to CoreML
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=(ct.RangeDim(1, 8192), obs_dim))],
            compute_units=ct.ComputeUnit.ALL,
        )

        return mlmodel

    except Exception as e:
        print(f"ANE conversion failed: {e}")
        return None


class ANEInference:
    """ANE inference wrapper."""

    def __init__(self, coreml_model):
        self.model = coreml_model

    def predict(self, obs_np):
        """Run inference."""
        result = self.model.predict({"obs": obs_np})
        return result[list(result.keys())[0]]
