from __future__ import annotations

from typing import Any, cast

import pytest

from unilab.base.run_control import RunComplete


def test_run_complete_summary_is_copied_and_read_only() -> None:
    source = {"collected_grasps": 2}
    completion = RunComplete(reason="grasp_collection_target_reached", summary=source)
    source["collected_grasps"] = 3

    assert str(completion) == "grasp_collection_target_reached"
    assert completion.reason == "grasp_collection_target_reached"
    assert completion.summary == {"collected_grasps": 2}
    with pytest.raises(TypeError):
        cast(Any, completion.summary)["collected_grasps"] = 4
    with pytest.raises(AttributeError):
        completion.summary = {}  # type: ignore[misc]
