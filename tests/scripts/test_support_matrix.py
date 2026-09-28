from __future__ import annotations

from pathlib import Path

import pytest
from scripts.tools.support_matrix import (
    BACKENDS,
    BEGIN_MARKER,
    END_MARKER,
    EntrypointSpec,
    EvidenceLevel,
    _configured_entries,
    _load_task_owner,
    build_support_rows,
)
from unisim.adapters import ADAPTER_SPECS
from unisim.capabilities import SupportLevel
from unisim.support import get_tensor_platform_profiles

ROOT = Path(__file__).resolve().parents[2]


def test_superdex_fr3_is_configured_without_full_training_claim() -> None:
    row = _row("PPO (torch)", "fr3_joint_target")
    assert row.cells["superdex"].level == EvidenceLevel.CONFIGURED
    go2_row = _row("PPO (torch)", "go2_joystick_flat")
    assert go2_row.cells["superdex"].level == EvidenceLevel.CONFIGURED


# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow


def _row(entrypoint_label: str, task_slug: str):
    for row in build_support_rows(ROOT):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def _generated_block(language: str) -> str:
    language_dirs = {"en": "en", "zh": "zh_CN"}
    path = (
        ROOT
        / "docs"
        / "sphinx"
        / "source"
        / language_dirs[language]
        / "5-reference"
        / "5-support_matrix.md"
    )
    content = path.read_text(encoding="utf-8")
    begin = content.index(BEGIN_MARKER)
    end = content.index(END_MARKER)
    return content[begin : end + len(END_MARKER)]


def _section(block: str, title: str) -> list[str]:
    lines = block.splitlines()
    start = lines.index(f"### {title}")
    try:
        end = next(
            index
            for index, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.startswith("### ")
        )
    except StopIteration:
        end = len(lines)
    return lines[start + 1 : end]


def _table_rows(section: list[str]) -> list[list[str]]:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section
        if line.lstrip().startswith("|")
    ]
    assert rows
    return rows


def test_support_matrix_marks_go2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "go2_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["mjwarp"].level == EvidenceLevel.MISSING
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_validated_g1_mjwarp_entrypoints_as_tested():
    torch_row = _row("PPO (torch)", "g1_walk_flat")
    sac_row = _row("SAC (torch)", "g1_walk_flat")

    assert torch_row.cells["mjwarp"].level == EvidenceLevel.TESTED
    assert sac_row.cells["mjwarp"].level == EvidenceLevel.TESTED


def test_backend_order_and_identity_follow_authoritative_unisim_profiles():
    """The matrix must not invent or reorder adapter identities."""
    declared = tuple(spec.name for spec in ADAPTER_SPECS)
    profiles = get_tensor_platform_profiles()

    assert BACKENDS == declared == tuple(profiles)
    assert len(BACKENDS) == len(set(BACKENDS))
    assert all(profiles[backend].adapter == backend for backend in BACKENDS)
    assert all(
        isinstance(
            getattr(profiles[backend], field),
            SupportLevel,
        )
        for backend in BACKENDS
        for field in (
            "reset_randomization",
            "fixed_variants",
            "host_pre_step_control",
            "packed_host_bridge",
        )
    )


def test_generator_renders_hardened_support_levels_without_enum_leakage():
    from scripts.generate_support_matrix import render_generated_block

    en_block = render_generated_block(ROOT, "en")
    zh_block = render_generated_block(ROOT, "zh")

    assert "SupportLevel." not in en_block
    assert "SupportLevel." not in zh_block
    assert "`mujoco`" in en_block
    mujoco_en = next(line for line in en_block.splitlines() if line.startswith("| `mujoco` |"))
    mujoco_zh = next(line for line in zh_block.splitlines() if line.startswith("| `mujoco` |"))
    assert mujoco_en.endswith("| unknown | unknown | Unsupported | Exact |")
    assert mujoco_zh.endswith("| unknown | unknown | 不支持 | 支持 |")


def test_owner_backend_identity_comes_from_training_sim_backend(
    tmp_path: Path,
) -> None:
    """A filename such as ``tensor`` must never disguise the selected adapter."""

    task_dir = tmp_path / "conf" / "demo-task"
    task_dir.mkdir(parents=True)
    owner = task_dir / "tensor.yaml"
    owner.write_text(
        "\n".join(
            (
                "training:",
                "  task_name: DemoTask",
                "  sim_backend: genesis",
                "env:",
                "  tensor_runtime: true",
            )
        ),
        encoding="utf-8",
    )
    (task_dir / "base.yaml").write_text("training: {}\n", encoding="utf-8")
    spec = EntrypointSpec(
        entrypoint_id="demo",
        label="Demo",
        config_dir="conf",
        task_glob="*/*.yaml",
    )

    assert _load_task_owner(owner) == ("DemoTask", "genesis", True)
    assert _configured_entries(tmp_path, spec) == {"demo-task": {"genesis": "DemoTask"}}


def test_platform_profiles_keep_host_and_device_boundaries_fail_closed() -> None:
    profiles = get_tensor_platform_profiles()

    for backend in ("mujoco", "motrix", "drake", "superdex"):
        profile = profiles[backend]
        assert profile.execution.value == "host_bridge"
        assert profile.process_topology.value == "in_process"
        assert profile.data_plane.value == "host_bridge"
        assert profile.torch_devices == ("cpu", "cuda")

    for backend in ("mjwarp", "newton", "genesis"):
        profile = profiles[backend]
        assert profile.execution.value == "device_resident"
        assert profile.process_topology.value == "in_process"
        assert profile.data_plane.value == "direct"
        assert profile.torch_devices == ("cuda",)

    for backend in ("isaacgym", "isaacsim"):
        profile = profiles[backend]
        assert profile.execution.value == "device_resident"
        assert profile.process_topology.value == "external_worker"
        assert profile.data_plane.value == "cuda_ipc"
        assert profile.torch_devices == ("cuda",)
        assert "host Python paths are not inherited" in profile.worker_requirement

    assert profiles["mujoco"].reset_randomization is SupportLevel.UNKNOWN
    assert profiles["mujoco"].fixed_variants is SupportLevel.UNKNOWN
    for backend in ("mjwarp", "newton", "genesis", "isaacgym", "isaacsim"):
        assert profiles[backend].reset_randomization is SupportLevel.UNSUPPORTED
    for backend in ("motrix", "drake", "superdex", "isaacgym", "isaacsim"):
        assert profiles[backend].fixed_variants is SupportLevel.UNSUPPORTED


def test_bilingual_generated_blocks_are_structurally_consistent():
    """Localization may differ; generated table structure and support data may not."""
    en_block = _generated_block("en")
    zh_block = _generated_block("zh")
    en_headings = tuple(line for line in en_block.splitlines() if line.startswith("### "))
    zh_headings = tuple(line for line in zh_block.splitlines() if line.startswith("### "))
    assert en_headings == zh_headings

    en_evidence = _table_rows(_section(en_block, "Evidence Grades"))
    zh_evidence = _table_rows(_section(zh_block, "Evidence Grades"))
    assert [row[0] for row in en_evidence[2:]] == [row[0] for row in zh_evidence[2:]]
    assert [len(row) for row in en_evidence] == [len(row) for row in zh_evidence]

    en_platform = _table_rows(_section(en_block, "Tensor Backend Platform Matrix"))
    zh_platform = _table_rows(_section(zh_block, "Tensor Backend Platform Matrix"))
    assert en_platform[0] == zh_platform[0]
    assert [row[0] for row in en_platform] == [row[0] for row in zh_platform]
    assert [row[:8] for row in en_platform] == [row[:8] for row in zh_platform]
    assert [len(row) for row in en_platform] == [len(row) for row in zh_platform]

    en_owners = _table_rows(_section(en_block, "Entrypoint x Task Owner"))
    zh_owners = _table_rows(_section(zh_block, "Entrypoint x Task Owner"))
    assert en_owners == zh_owners

    assert _section(en_block, "Source Index") == _section(zh_block, "Source Index")


def test_doc_checks_validate_current_generated_blocks_for_both_languages():
    from tests.scripts.doc_checks import check_generated_support_matrix

    language_dirs = {"en": "en", "zh": "zh_CN"}
    for language, language_dir in language_dirs.items():
        path = (
            ROOT
            / "docs"
            / "sphinx"
            / "source"
            / language_dir
            / "5-reference"
            / "5-support_matrix.md"
        )
        assert check_generated_support_matrix(path.read_text(encoding="utf-8"), path, ROOT) == []


def test_support_matrix_marks_g1_isaacgym_owners_by_validation():
    """SAC isaacgym is maintainer-validated on hardware; PPO stays CONFIGURED."""
    sac_row = _row("SAC (torch)", "g1_walk_flat")
    assert sac_row.cells["isaacgym"].level == EvidenceLevel.TESTED
    ppo_row = _row("PPO (torch)", "g1_walk_flat")
    assert ppo_row.cells["isaacgym"].level == EvidenceLevel.CONFIGURED
    for entrypoint_label in (
        "APPO (torch)",
        "FlashSAC (torch)",
    ):
        row = _row(entrypoint_label, "g1_walk_flat")
        # Registration is per task+backend, not per algo tree: without an
        # owner YAML these stay at REGISTERED instead of CONFIGURED.
        assert row.cells["isaacgym"].level == EvidenceLevel.REGISTERED


def test_support_matrix_marks_g1_isaacsim_owners_by_checked_in_scope():
    """IsaacSim has owner YAMLs for PPO/SAC but no maintainer training claim."""
    ppo_row = _row("PPO (torch)", "g1_walk_flat")
    sac_row = _row("SAC (torch)", "g1_walk_flat")
    assert ppo_row.cells["isaacsim"].level == EvidenceLevel.CONFIGURED
    assert sac_row.cells["isaacsim"].level == EvidenceLevel.CONFIGURED
    for entrypoint_label in ("APPO (torch)", "FlashSAC (torch)"):
        row = _row(entrypoint_label, "g1_walk_flat")
        assert row.cells["isaacsim"].level == EvidenceLevel.REGISTERED


def test_support_matrix_marks_g1_newton_owner_sac_tested_ppo_configured():
    """SAC newton is training-validated (Tested); PPO stays Configured."""
    row = _row("SAC (torch)", "g1_walk_flat")
    assert row.cells["newton"].level == EvidenceLevel.TESTED
    row = _row("PPO (torch)", "g1_walk_flat")
    assert row.cells["newton"].level == EvidenceLevel.CONFIGURED
    for entrypoint_label in ("APPO (torch)", "FlashSAC (torch)"):
        row = _row(entrypoint_label, "g1_walk_flat")
        # Registration is per task+backend, not per algo tree: without an
        # owner YAML these stay at REGISTERED instead of CONFIGURED.
        assert row.cells["newton"].level == EvidenceLevel.REGISTERED


def test_support_matrix_does_not_promote_unvalidated_newton_entries():
    rows = build_support_rows(Path(__file__).resolve().parents[2])

    tested = {
        (row.entrypoint_label, row.task_slug)
        for row in rows
        if row.cells["newton"].level >= EvidenceLevel.TESTED
    }
    assert tested == {("SAC (torch)", "g1_walk_flat")}
    go2_row = _row("PPO (torch)", "go2_joystick_flat")
    assert go2_row.cells["newton"].level == EvidenceLevel.MISSING


def test_support_matrix_does_not_promote_unvalidated_isaacgym_entries():
    rows = build_support_rows(Path(__file__).resolve().parents[2])

    tested = {
        (row.entrypoint_label, row.task_slug)
        for row in rows
        if row.cells["isaacgym"].level >= EvidenceLevel.TESTED
    }
    assert tested == {("SAC (torch)", "g1_walk_flat")}
    go2_row = _row("PPO (torch)", "go2_joystick_flat")
    assert go2_row.cells["isaacgym"].level == EvidenceLevel.MISSING


def test_issue_gated_motion_tracking_owners_do_not_promote_support_cells():
    """M9 candidate benchmark owners remain outside the public support matrix."""
    row = _row("FlashSAC (torch)", "g1_motion_tracking")

    assert row.cells["isaacgym"].level == EvidenceLevel.REGISTERED
    assert row.cells["isaacsim"].level == EvidenceLevel.REGISTERED


def test_support_matrix_marks_g1_genesis_owner_configured_only():
    """SAC genesis is training-validated (Tested); PPO stays Configured."""
    row = _row("SAC (torch)", "g1_walk_flat")
    assert row.cells["genesis"].level == EvidenceLevel.TESTED
    row = _row("PPO (torch)", "g1_walk_flat")
    assert row.cells["genesis"].level == EvidenceLevel.CONFIGURED
    for entrypoint_label in (
        "APPO (torch)",
        "FlashSAC (torch)",
    ):
        row = _row(entrypoint_label, "g1_walk_flat")
        # Registration is per task+backend, not per algo tree: without an
        # owner YAML these stay at REGISTERED instead of CONFIGURED.
        assert row.cells["genesis"].level == EvidenceLevel.REGISTERED


def test_support_matrix_does_not_promote_unvalidated_genesis_entries():
    rows = build_support_rows(Path(__file__).resolve().parents[2])

    tested = {
        (row.entrypoint_label, row.task_slug)
        for row in rows
        if row.cells["genesis"].level >= EvidenceLevel.TESTED
    }
    assert tested == {("SAC (torch)", "g1_walk_flat")}
    go2_row = _row("PPO (torch)", "go2_joystick_flat")
    assert go2_row.cells["genesis"].level == EvidenceLevel.MISSING


def test_support_matrix_does_not_promote_unvalidated_mjwarp_entries():
    rows = build_support_rows(Path(__file__).resolve().parents[2])

    tested = {
        (row.entrypoint_label, row.task_slug)
        for row in rows
        if row.cells["mjwarp"].level >= EvidenceLevel.TESTED
    }
    assert tested == {
        ("PPO (torch)", "g1_walk_flat"),
        ("SAC (torch)", "g1_walk_flat"),
        ("WarpSAC (torch)", "g1_walk_flat"),
        ("WarpSAC (torch)", "g1_motion_tracking"),
    }
    appo_row = _row("APPO (torch)", "g1_walk_flat")
    assert appo_row.cells["mjwarp"].level == EvidenceLevel.REGISTERED


def test_support_matrix_marks_allegro_appo_backends_as_tested():
    allegro_appo_row = _row("APPO (torch)", "allegro_inhand")

    assert allegro_appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert allegro_appo_row.cells["motrix"].level == EvidenceLevel.TESTED
