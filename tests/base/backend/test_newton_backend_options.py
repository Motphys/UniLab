import pytest

from unilab.base.backend_factory import env_backend_kwargs
from unilab.base.base import EnvCfg


def test_newton_cuda_graph_option_defaults_to_eager() -> None:
    cfg = EnvCfg()
    cfg.validate()

    assert cfg.newton_use_cuda_graph is False
    assert "newton_use_cuda_graph" not in env_backend_kwargs(cfg)


def test_newton_cuda_graph_option_routes_to_backend_factory() -> None:
    cfg = EnvCfg(newton_use_cuda_graph=True)
    cfg.validate()

    assert env_backend_kwargs(cfg)["newton_use_cuda_graph"] is True


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_newton_cuda_graph_option_requires_bool(value: object) -> None:
    cfg = EnvCfg(newton_use_cuda_graph=value)

    with pytest.raises(ValueError, match="newton_use_cuda_graph must be bool"):
        cfg.validate()
