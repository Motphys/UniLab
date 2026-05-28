# fast_td3

::::{admonition} TODO
:class: note
This page is a stub. PRs welcome. The implementation lives in
{py:mod}`unilab.algos`; see the API reference for the current interface.
::::

## Quick start

```bash
uv run train --algo fast_td3 --task <task> --sim <backend>
```

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo td3`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`../getting_started/evaluation_and_playback`.

## See also

- {doc}`../algorithms/overview`
- {doc}`../../api_reference/algos/index`
