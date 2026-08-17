"""ONNX export and onnxruntime numeric verification helpers for play entrypoints."""

from __future__ import annotations

import numpy as np
import torch


def export_policy_onnx(
    export_module: torch.nn.Module,
    onnx_path: str,
    export_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str] | None = None,
    opset_version: int = 17,
) -> None:
    """Export ``export_module`` to ``onnx_path`` and print the artifact path.

    Args:
        export_module: Module traced by ``torch.onnx.export``.
        onnx_path: Destination file path for the exported graph.
        export_inputs: Positional example inputs matching ``input_names``.
        input_names: ONNX input names, aligned positionally with ``export_inputs``.
        output_names: ONNX output names; defaults to ``["action"]``.
        opset_version: ONNX opset version; defaults to 17.
    """
    if output_names is None:
        output_names = ["action"]
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            export_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
        )
    print(f"Exported actor ONNX to {onnx_path}")


def verify_policy_onnx(
    export_module: torch.nn.Module,
    onnx_path: str,
    verify_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    max_diff_tol: float = 1e-4,
) -> tuple[float, float]:
    """Compare PyTorch and ONNX Runtime outputs on identical inputs.

    Runs ``export_module`` and the exported graph at ``onnx_path`` on
    ``verify_inputs``, prints the max/mean absolute difference, and prints a
    warning when the max difference exceeds ``max_diff_tol``.

    Args:
        export_module: Module that was exported to ``onnx_path``. Tuple outputs
            are compared on their first element.
        onnx_path: Exported ONNX graph to verify.
        verify_inputs: Positional inputs matching ``input_names``.
        input_names: ONNX input names, aligned positionally with ``verify_inputs``.
        max_diff_tol: Absolute max-difference tolerance before warning.

    Returns:
        ``(max_diff, mean_diff)`` between PyTorch and ONNX Runtime outputs.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        pt_output = export_module(*verify_inputs)
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0]
        pt_np = pt_output.cpu().numpy()
    onnx_inputs = {
        name: value.cpu().numpy().astype(np.float32)
        for name, value in zip(input_names, verify_inputs, strict=True)
    }
    onnx_output = sess.run(None, onnx_inputs)[0]
    max_diff = float(np.max(np.abs(pt_np - onnx_output)))
    mean_diff = float(np.mean(np.abs(pt_np - onnx_output)))
    print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
    if max_diff > max_diff_tol:
        print("WARNING: ONNX output diverges from PyTorch!")
    else:
        print("ONNX export verified OK.")
    return max_diff, mean_diff
