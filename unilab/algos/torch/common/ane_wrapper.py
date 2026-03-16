"""ANE (Apple Neural Engine) inference wrapper for Fast SAC."""

import numpy as np
import torch


class ANEActorWrapper:
    """Wrapper to run actor inference on Apple Neural Engine via CoreML."""

    def __init__(self, actor_model, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.coreml_model = None

        try:
            import coremltools as ct

            # Convert PyTorch model to CoreML
            example_input = torch.randn(1, obs_dim)
            traced_model = torch.jit.trace(actor_model, example_input)

            # Convert to CoreML with ANE optimization
            self.coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=(ct.RangeDim(1, 8192), obs_dim))],
                compute_units=ct.ComputeUnit.ALL,  # Use ANE when available
                minimum_deployment_target=ct.target.macOS13,
            )
            print("✓ ANE model converted successfully")

        except ImportError:
            raise ImportError("coremltools not installed. Run: pip install coremltools")
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to CoreML: {e}")

    def predict(self, obs_np):
        """Run inference on ANE.

        Args:
            obs_np: numpy array of shape (batch, obs_dim)

        Returns:
            actions: numpy array of shape (batch, action_dim)
        """
        if self.coreml_model is None:
            raise RuntimeError("CoreML model not initialized")

        # CoreML expects dict input
        result = self.coreml_model.predict({"input": obs_np})

        # Extract output (key depends on model structure)
        output_key = list(result.keys())[0]
        actions = result[output_key]

        return actions

    def __call__(self, obs_np):
        return self.predict(obs_np)
