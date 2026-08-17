from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest


def test_him_iteration_progress_is_one_equivalent_multiline_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner

    runner = cast(Any, HIMOnPolicyRunner.__new__(HIMOnPolicyRunner))
    runner.logger = SimpleNamespace(
        rewbuffer=deque([1.0, 3.0]),
        lenbuffer=deque([10.0, 14.0]),
    )

    with caplog.at_level(logging.INFO, logger="unilab.algos.torch.him_ppo.runner"):
        runner._log_iter(
            it=1,
            tot=3,
            value_loss=0.25,
            surrogate_loss=0.5,
            estimation_loss=0.75,
            swap_loss=1.0,
            elapsed=2.0,
            infos={"log": {"reward/feet": 1.25}},
        )

    records = [
        record for record in caplog.records if record.name == "unilab.algos.torch.him_ppo.runner"
    ]
    assert len(records) == 1
    lines = records[0].getMessage().splitlines()
    assert lines[0] == "-" * 80
    assert lines[-1] == "-" * 80
    assert f"{'Iteration':>40}: 1/3" in lines
    assert f"{'Mean episode reward':>40}: 2.0000" in lines
    assert f"{'Mean episode length':>40}: 12.0" in lines
    assert f"{'reward/feet':>40}: 1.2500" in lines
    assert f"{'Time elapsed':>40}: 00:00:02" in lines
    assert f"{'ETA':>40}: 00:00:04" in lines
