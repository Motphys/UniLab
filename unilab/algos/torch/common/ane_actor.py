"""ANE actor for Fast SAC - correct implementation."""

import tempfile

import coremltools as ct
import numpy as np
import torch


class ANEActor:
    """Actor using Apple Neural Engine."""

    def __init__(self, actor_model, obs_dim, action_dim, batch_sizes=[4096]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mlmodel = self._convert_to_coreml(actor_model, obs_dim, batch_sizes)
        self.input_name = list(self.mlmodel.get_spec().description.input)[0].name

    def _convert_to_coreml(self, actor, obs_dim, batch_sizes):
        """Convert PyTorch actor to CoreML with EnumeratedShapes."""

        # Wrapper for deterministic mean action
        class MeanActor(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                mean, _ = self.actor.forward(obs)
                return torch.tanh(mean)

        wrapper = MeanActor(actor)
        wrapper.eval()

        # Trace model
        example = torch.randn(batch_sizes[0], obs_dim)
        traced = torch.jit.trace(wrapper, example)

        # Convert with EnumeratedShapes for ANE
        shapes = [[n, obs_dim] for n in batch_sizes]
        input_shape = ct.EnumeratedShapes(shapes=shapes, default=shapes[-1])

        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="obs", shape=input_shape)],
            compute_precision=ct.precision.FLOAT16,
        )

        # Save to temp file and reload with ANE
        temp_path = tempfile.mktemp(suffix=".mlpackage")
        mlmodel.save(temp_path)

        return ct.models.MLModel(temp_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    def explore(self, obs_np, deterministic=False):
        """Inference with exploration noise."""
        result = self.mlmodel.predict({self.input_name: obs_np})
        actions = result[list(result.keys())[0]]

        if not deterministic:
            noise = np.random.normal(0, 0.1, actions.shape).astype(np.float32)
            actions = np.clip(actions + noise, -1, 1)

        return actions
