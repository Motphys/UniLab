from __future__ import annotations

import ast
import inspect
import textwrap


def test_hora_rsl_wrapper_uses_explicit_np_env_state_contract() -> None:
    """HORA wrapper must not probe required NpEnvState fields dynamically."""
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    source = textwrap.dedent(inspect.getsource(HoraRslRlVecEnvWrapper.step))
    tree = ast.parse(source)
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        if node.args[0].id == "state":
            forbidden_calls.append(node.func.id)

    assert forbidden_calls == []
