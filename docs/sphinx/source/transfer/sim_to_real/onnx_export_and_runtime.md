# ONNX Export and Runtime

A UniLab policy lives inside a `torch.nn.Module` (or an MLX module on Mac).
To put it on a robot you need a **frozen, language-neutral graph** plus the
**observation statistics** that were used in training. ONNX is the format
UniLab targets for hardware deployment.

## What gets exported

```{list-table}
:header-rows: 1
:widths: 25 75

* - Artefact
  - Notes
* - `actor.onnx`
  - Forward graph of the policy actor only. Critic / discriminator are not
    needed on hardware.
* - `obs_stats.npz`
  - Running mean & variance per observation group, captured from the
    training run.
* - `action_scale.npz`
  - Per-joint action range and zero-offset (from the env config).
* - `manifest.yaml`
  - Plain-text record of `(task, backend, algo, git sha, train command)`.
* - (optional) `actor.mlpackage`
  - Apple Neural Engine package, produced from MLX or via the ANE wrapper.
```

## Standard export path (PyTorch policies)

UniLab ships an actor wrapper specifically designed to be ONNX-clean (no
in-place ops, no Python control flow, single forward signature). The
relevant module is {py:mod}`unilab.algos.torch.common.ane_wrapper`.

```python
import torch
from unilab.algos.torch.common.ane_actor import build_ane_actor
from unilab.algos.torch.common.ane_wrapper import wrap_for_onnx

actor = build_ane_actor(checkpoint_path="runs/go2_joystick_flat/.../model_5000.pt")
onnx_actor = wrap_for_onnx(actor)

example_obs = torch.zeros(1, actor.obs_dim, dtype=torch.float32)
torch.onnx.export(
    onnx_actor,
    example_obs,
    "actor.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=17,
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
)
```

::::{admonition} Why batch dim?
:class: tip
Even on a single robot you often run **multiple controllers concurrently**
(low-level whole-body + high-level intent). Keeping the batch axis dynamic
lets ONNX runtime fuse them into one inference call.
::::

## Apple Neural Engine path

On macOS / iOS deployments, prefer the ANE pipeline:

```python
from unilab.algos.torch.common.ane_inference import compile_for_ane

ane_actor = compile_for_ane(
    actor, example_obs, output_dir="runs/.../ane"
)
```

This converts via Core ML and emits a `.mlpackage` you can `coremlc compile`
on-device. See {py:mod}`unilab.algos.torch.common.ane_inference` for the
exact knobs (precision, compute units, target deployment target).

## Verifying the exported graph

The number-one debugging tactic: run **`actor.onnx` and the original
PyTorch actor on the same 100 observations** and assert `|Δ| < 1e-4`.

```bash
uv run python - <<'PY'
import numpy as np, onnxruntime as ort, torch
from unilab.algos.torch.common.ane_actor import build_ane_actor

actor = build_ane_actor("runs/.../model_5000.pt").eval()
sess  = ort.InferenceSession("actor.onnx", providers=["CPUExecutionProvider"])

rng = np.random.default_rng(0)
for _ in range(100):
    obs = rng.standard_normal((1, actor.obs_dim)).astype(np.float32)
    a_pt  = actor(torch.from_numpy(obs)).detach().cpu().numpy()
    a_ort = sess.run(["action"], {"obs": obs})[0]
    assert np.max(np.abs(a_pt - a_ort)) < 1e-4
print("OK")
PY
```

If the assertion fails on a few obs only, you almost certainly have a
non-deterministic op (dropout left on, LayerNorm in train mode). Re-run
with `.eval()`.

## Where to keep export scripts

UniLab does not ship a global `unilab export-onnx` command on purpose — the
right export depends on the actor architecture. Each task owner that has
real-hardware coverage carries its own `scripts/export_<task>.py`. Pattern
your robot's script after the closest existing one:

| Robot | Reference script |
|---|---|
| G1 humanoid | `scripts/export_g1_motion_tracking.py` |
| Go2 / Go2W | `scripts/export_go2_joystick.py` |
| Allegro / Sharpa | `scripts/export_inhand.py` |

## See also

- {doc}`latency_and_observation_lag`
- {doc}`safety_layers`
- {py:class}`unilab.algos.torch.common.ane_actor.ANEActor`
