from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from tests.scripts import doc_checks


def test_documentation_files_match_current_repo_contracts():
    root = Path(__file__).resolve().parents[2]
    errors = doc_checks.collect_doc_errors(root)
    assert errors == []


def test_sharpa_domain_randomization_doc_matches_owner_config():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "docs" / "users" / "zh_CN" / "06-domain-randomization.md"
    content = doc_path.read_text(encoding="utf-8")
    owner_cfg = OmegaConf.load(root / "conf" / "ppo" / "task" / "sharpa_inhand" / "mujoco.yaml")

    sharpa_rows = [
        line for line in content.splitlines() if line.startswith("| `SharpaInhandRotation` |")
    ]
    assert sharpa_rows
    if float(owner_cfg.env.domain_rand.force_scale) > 0.0:
        assert all("| 无 |" not in row for row in sharpa_rows)
        assert any("body_force" in row for row in sharpa_rows)

    assert "`env.domain_rand.scale_list`" in content
    assert "`env.scale_list`" not in content
    assert "必须能被 `num_scales` 整除" not in content


def test_check_training_entrypoint_semantics_flags_issue_204_patterns():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "README.md"
    content = """
uv run scripts/train_rsl_rl.py task=go1_joystick_flat
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco training.load_run=2026-01-01
Training logs are saved to logs/rsl_rl_train/MyTask/.
"""

    errors = doc_checks.check_training_entrypoint_semantics(content, doc_path, root)

    assert any("training.load_run" in error for error in errors)
    assert any("logs/rsl_rl_train/" in error for error in errors)
    assert any("task=go1_joystick_flat" in error for error in errors)


def test_check_training_entrypoint_semantics_accepts_current_patterns():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "README.md"
    content = """
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco algo.load_run=2026-01-01
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
Logs live under logs/<algo.algo_log_name>/<task>/.
"""

    errors = doc_checks.check_training_entrypoint_semantics(content, doc_path, root)

    assert errors == []


def test_check_script_references_ignores_tests_paths():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "CONTRIBUTING.md"
    content = "Run uv run pytest tests/scripts/test_check_docs.py -q"

    errors = doc_checks.check_script_references(content, doc_path, root)

    assert errors == []


def test_collect_doc_errors_scans_issue_templates_for_hydra_semantics(tmp_path):
    issue_template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    issue_template.parent.mkdir(parents=True)
    issue_template.write_text(
        "placeholder: |\n  uv run scripts/train_offpolicy.py algo=sac task=g1_walk_flat ...\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "train_offpolicy.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("task=g1_walk_flat" in error for error in errors)


def test_collect_doc_errors_scans_issue_templates_for_script_paths(tmp_path):
    issue_template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    issue_template.parent.mkdir(parents=True)
    issue_template.write_text(
        "placeholder: |\n  uv run scripts/missing_entrypoint.py task=go1_joystick_flat/mujoco\n",
        encoding="utf-8",
    )

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("Script not found: scripts/missing_entrypoint.py" in error for error in errors)


def test_collect_doc_errors_flags_unclosed_markdown_fence(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```bash\nuv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco\n", encoding="utf-8"
    )

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("Unclosed fenced code block" in error for error in errors)


def test_collect_doc_errors_flags_python_script_invocation(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```bash\nuv run python scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco\n```\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "train_rsl_rl.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("Use `uv run scripts/...`" in error for error in errors)


def test_collect_doc_errors_scans_scripts_markdown(tmp_path):
    doc_path = tmp_path / "scripts" / "motion" / "README.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        "```bash\nuv run python scripts/motion/replay_npz.py\n```\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "motion" / "replay_npz.py"
    script_path.write_text("", encoding="utf-8")

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("Use `uv run scripts/...`" in error for error in errors)


def test_check_zh_cn_doc_shape_requires_language_navigation_and_index(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "users" / "zh_CN" / "06-domain-randomization.md"
    doc_path.parent.mkdir(parents=True)
    content = "# 域随机化\n\n缺少语言头。\n"

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert any("语言: 简体中文" in error for error in errors)
    assert any("## Navigation" in error for error in errors)
    assert any("docs/README.md" in error for error in errors)


def test_check_zh_cn_doc_shape_accepts_user_contract(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "users" / "zh_CN" / "06-domain-randomization.md"
    doc_path.parent.mkdir(parents=True)
    content = (
        "# 域随机化\n\n语言: 简体中文\n\n正文。\n\n"
        "## Navigation\n\n- Index: [Documentation](../../README.md)\n"
    )

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert errors == []


def test_check_zh_cn_doc_shape_accepts_developer_contract(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "developers" / "zh_CN" / "development-standard.md"
    doc_path.parent.mkdir(parents=True)
    content = (
        "# RL Infrastructure 开发标准\n\n语言: 简体中文\n\n正文。\n\n"
        "## Navigation\n\n- Index: [Documentation](../../README.md)\n"
    )

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert errors == []


def test_check_adr_shape_requires_governance_fields(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "developers" / "adr" / "ADR-9999-example.md"
    doc_path.parent.mkdir(parents=True)
    content = "# ADR-9999 Example\n\n- Status: Accepted\n"

    errors = doc_checks.check_adr_shape(content, doc_path, root)

    assert any("Supersedes" in error for error in errors)
    assert any("Alternatives Considered" in error for error in errors)
    assert any("Evidence In Repo" in error for error in errors)


def test_check_adr_shape_accepts_template_contract(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "developers" / "adr" / "ADR-9999-example.md"
    doc_path.parent.mkdir(parents=True)
    content = """
# ADR-9999 Example

- Status: Accepted
- Date: 2026-05-04
- Owners: Maintainers
- Supersedes: None
- Superseded by: None

## Alternatives Considered

None.

## Evidence In Repo

- tests/example.py

## Related Documents

- [ADR Index](README.md)
"""

    errors = doc_checks.check_adr_shape(content, doc_path, root)

    assert errors == []


def test_collect_doc_errors_flags_removed_unilab_subcommands(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```bash\nsource .venv/bin/activate\nunilab train --algo ppo --task go2 --sim mujoco\n```\n",
        encoding="utf-8",
    )

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("removed `unilab` subcommand interface" in error for error in errors)
    assert any("instead of activating .venv" in error for error in errors)


def test_collect_doc_errors_flags_broken_raw_github_repo_url(tmp_path):
    issue_template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    issue_template.parent.mkdir(parents=True)
    issue_template.write_text(
        "url: https://github.com/unilabsim/UniLab/blob/main/docs/missing.md\n",
        encoding="utf-8",
    )

    errors = doc_checks.collect_doc_errors(tmp_path)

    assert any("GitHub URL target not found: docs/missing.md" in error for error in errors)
