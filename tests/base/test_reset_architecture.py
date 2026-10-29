import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "unilab"


def test_reset_chain_does_not_materialize_dynamic_done_rows() -> None:
    guarded_files = (
        "envs/manager_based_rl_env.py",
        "managers/event_manager.py",
        "managers/command_manager.py",
        "managers/observation_manager.py",
        "rl/adapters/rsl_rl/vec_env.py",
    )
    forbidden = (
        "reset_buf.nonzero",
        "terminated.nonzero",
        "truncated.nonzero",
        "dones.nonzero",
        "done.nonzero",
    )
    for relative_path in guarded_files:
        source = (SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), relative_path


def test_public_reset_boundaries_speak_env_ids() -> None:
    """The public reset contract is the IsaacLab/mjlab ``env_ids`` row-index form."""
    checked = (
        ("base/torch_env.py", "reset"),
        ("managers/manager_base.py", "reset"),
        ("managers/event_manager.py", "reset"),
        ("managers/event_manager.py", "apply"),
    )
    for relative_path, function in checked:
        tree = ast.parse((SOURCE_ROOT / relative_path).read_text(encoding="utf-8"))
        signatures = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
        ]
        assert signatures, (relative_path, function)
        for node in signatures:
            arg_names = {
                arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            }
            assert "env_ids" in arg_names, (relative_path, function)
            assert "reset_mask" not in arg_names, (relative_path, function)
    event_source = (SOURCE_ROOT / "managers" / "event_manager.py").read_text(encoding="utf-8")
    assert "min_step_count_between_reset" in event_source


def test_removed_reset_row_contracts_stay_removed() -> None:
    removed = (
        "uses_tensor_reset_rows",
        "reset_command_state",
        "compute_reset_rows",
        "compute_reset",
        "update_command_reset",
        "reset_to_default_mask",
        "write_root_state_tensor_to_sim_mask",
        "write_motion_state_tensor_to_sim_mask",
        "commit_tensor(",
        "can_commit_packed",
        "refresh_after_selected_reset",
        "_reset_tensor_staging_mask",
        "_reset_tensor_staging_rows",
        "_tensor_state_staging",
        "_commit_mocap_poses",
        "_build_randomization_payload",
        "_fill_sparse_randomization_rows",
        "_fill_sparse_gain_rows",
        "_record_committed_payload",
        "_promote_packed_reset_baseline",
        "_promote_startup_randomization_defaults",
        "_host_randomization_staged",
        "_materialize_tensor_default_state",
        "_packed_reset_widths_match",
        "_packed_reset_public_widths",
        "_packed_reset_plane",
        "_interval_capability_plane",
        "_tensor_randomization_columns",
        "_validate_gain_vector",
        "_canonical_default_shape",
        "_randomization_default_rows",
        "_mask_full_refresh",
    )
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        identifiers.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        identifiers.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        assert not identifiers.intersection(removed), path
