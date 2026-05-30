from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from tests.scripts import doc_checks


def test_documentation_files_match_current_repo_contracts():
    root = Path(__file__).resolve().parents[2]
    errors = doc_checks.collect_doc_errors(root)
    assert errors == []
    warnings = doc_checks.collect_doc_warnings(root)
    assert warnings == []


def test_sharpa_domain_randomization_doc_matches_owner_config():
    root = Path(__file__).resolve().parents[2]
    doc_path = (
        root / "docs" / "sphinx" / "source" / "zh_CN" / "1-user_guide" / "5-domain-randomization.md"
    )
    content = doc_path.read_text(encoding="utf-8")

    owner_cfg = OmegaConf.load(root / "conf" / "ppo" / "task" / "sharpa_inhand" / "mujoco.yaml")

    assert "Sharpa" in content
    assert "`geom_size`" in content
    if float(owner_cfg.env.domain_rand.force_scale) > 0.0:
        assert "`gravity`" in content


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


def test_check_hydra_keys_ignores_non_command_fenced_blocks():
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "README.md"
    content = """
```bibtex
@article{jia2026unilab,
  title  = {UniLab},
  author = {UniLab Contributors},
  year   = {2026}
}
```

```bash
uv run scripts/train_rsl_rl.py missing_key=true task=go1_joystick_flat/mujoco
```
"""

    errors = doc_checks.check_hydra_keys(content, doc_path, root)

    assert not any("title" in error or "author" in error or "year" in error for error in errors)
    assert any("Unknown config key: missing_key" in error for error in errors)


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
    doc_path = (
        root / "docs" / "sphinx" / "source" / "zh_CN" / "1-user_guide" / "5-domain-randomization.md"
    )
    doc_path.parent.mkdir(parents=True)
    content = "# 域随机化\n\n缺少语言头。\n"

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert any("语言: 简体中文" in error for error in errors)
    assert any("## Navigation" in error for error in errors)
    assert any("docs index" in error for error in errors)


def test_check_zh_cn_doc_shape_accepts_user_contract(tmp_path):
    root = tmp_path
    doc_path = (
        root / "docs" / "sphinx" / "source" / "zh_CN" / "1-user_guide" / "5-domain-randomization.md"
    )
    doc_path.parent.mkdir(parents=True)
    content = (
        "# 域随机化\n\n语言: 简体中文\n\n正文。\n\n"
        "## Navigation\n\n- Index: [Documentation](../index.md)\n"
    )

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert errors == []


def test_check_zh_cn_doc_shape_accepts_developer_contract(tmp_path):
    root = tmp_path
    doc_path = (
        root
        / "docs"
        / "sphinx"
        / "source"
        / "zh_CN"
        / "2-developer_guide"
        / "1-development-standard.md"
    )
    doc_path.parent.mkdir(parents=True)
    content = (
        "# RL Infrastructure 开发标准\n\n语言: 简体中文\n\n正文。\n\n"
        "## Navigation\n\n- Index: [Documentation](../index.md)\n"
    )

    errors = doc_checks.check_zh_cn_doc_shape(content, doc_path, root)

    assert errors == []


def test_check_user_doc_architecture_flags_migration_phrasing_and_missing_task_sections(tmp_path):
    root = tmp_path
    task_index = (
        root
        / "docs"
        / "sphinx"
        / "source"
        / "zh_CN"
        / "1-user_guide"
        / "4-tasks"
        / "1-task-index.md"
    )
    task_index.parent.mkdir(parents=True)
    task_index.write_text("# 任务\n\n语言: 简体中文\n\n已拆成。\n", encoding="utf-8")

    from tests.scripts.doc_checks import check_user_doc_architecture

    errors = check_user_doc_architecture(task_index.read_text(encoding="utf-8"), task_index, root)

    assert any("已拆成" in error for error in errors)
    assert any("按机器人家族" in error for error in errors)
    assert any("按任务类型" in error for error in errors)


def test_check_user_doc_architecture_requires_backend_matrix_route(tmp_path):
    root = tmp_path
    doc_path = (
        root / "docs" / "sphinx" / "source" / "zh_CN" / "1-user_guide" / "2-simulation-backends.md"
    )
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 仿真后端\n\n语言: 简体中文\n\n正文。\n", encoding="utf-8")

    from tests.scripts.doc_checks import check_user_doc_architecture

    errors = check_user_doc_architecture(doc_path.read_text(encoding="utf-8"), doc_path, root)

    assert any("5-reference/1-backend-support-matrix.md" in error for error in errors)


def test_check_sphinx_source_migration_guards_flags_removed_paths_and_english_navigation(
    tmp_path,
):
    root = tmp_path
    doc_path = root / "docs" / "sphinx" / "source" / "en" / "user_guide" / "old.md"
    doc_path.parent.mkdir(parents=True)
    content = """
# Old Page

See [Documentation](../../README.md).
See docs/users/zh_CN/03-training.md and docs/developers/zh_CN/collaboration.md.
See ../../users/zh_CN/1-simulation-backends.md and ../../developers/adr/README.md.
语言: 简体中文

## Navigation
"""

    warnings = doc_checks.check_sphinx_source_migration_guards(content, doc_path, root)

    assert any("../../README.md" in warning for warning in warnings)
    assert any("docs/users/" in warning for warning in warnings)
    assert any("docs/developers/" in warning for warning in warnings)
    assert any("users/" in warning for warning in warnings)
    assert any("developers/" in warning for warning in warnings)
    assert any("语言: 简体中文" in warning for warning in warnings)
    assert any("English Sphinx pages" in warning for warning in warnings)


def test_collect_doc_warnings_scans_sphinx_source_migration_guards(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "sphinx" / "source" / "en" / "old.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("See [Documentation](../../README.md).\n", encoding="utf-8")

    warnings = doc_checks.collect_doc_warnings(root)

    assert any("../../README.md" in warning for warning in warnings)


def test_check_sphinx_source_migration_guards_accepts_current_doc_roles(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "sphinx" / "source" / "adr" / "README.md"
    doc_path.parent.mkdir(parents=True)
    content = """
# ADR

- {doc}`Documentation </index>`
    - {doc}`仿真后端 </zh_CN/1-user_guide/2-simulation-backends>`
- {doc}`ADR Index </adr/README>`
"""

    warnings = doc_checks.check_sphinx_source_migration_guards(content, doc_path, root)

    assert warnings == []


def test_check_adr_shape_requires_governance_fields(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "sphinx" / "source" / "adr" / "ADR-9999-example.md"
    doc_path.parent.mkdir(parents=True)
    content = "# ADR-9999 Example\n\n- Status: Accepted\n"

    errors = doc_checks.check_adr_shape(content, doc_path, root)

    assert any("Supersedes" in error for error in errors)
    assert any("Alternatives Considered" in error for error in errors)
    assert any("Evidence In Repo" in error for error in errors)


def test_check_adr_shape_accepts_template_contract(tmp_path):
    root = tmp_path
    doc_path = root / "docs" / "sphinx" / "source" / "adr" / "ADR-9999-example.md"
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

- [ADR Index](ADR-0000-index.md)
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
