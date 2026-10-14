from pathlib import Path

import pytest

from unilab import robot_arm_demo


def test_registered_robot_arm_demo_assets_exist() -> None:
    for demo_name in robot_arm_demo.ROBOT_ARM_DEMOS:
        path = robot_arm_demo.robot_model_path(demo_name)
        assert path.is_file()
        assert path.suffix == ".xml"


def test_unknown_robot_arm_demo_lists_available_names() -> None:
    with pytest.raises(ValueError, match="a1-left-arm"):
        robot_arm_demo.robot_model_path("missing")


def test_robot_model_path_fails_closed_when_asset_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(robot_arm_demo, "ASSETS_ROOT_PATH", tmp_path)
    with pytest.raises(FileNotFoundError, match="Robot-arm demo model not found"):
        robot_arm_demo.robot_model_path("a1-left-arm")
