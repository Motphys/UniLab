"""Env injection contracts and observation semantics (single owner).

Cold-path contract definitions only; this subpackage depends on stdlib,
typing, and Torch types, never on concrete envs, registries, or algorithms.
"""

from unilab.base.contracts.env import (
    EnvAlgoCapabilities,
    EnvAlgoCapabilitiesProtocol,
    EnvFactory,
    EnvPlayCapabilitiesProtocol,
    EnvProtocol,
    EnvStateProtocol,
    SupportsAlgoCapabilitiesProtocol,
    SupportsBackendDiagnosticsProtocol,
    SupportsCompletionReceiptsProtocol,
    get_algo_capabilities,
    get_backend_runtime_diagnostics,
    get_produces_completion_receipts,
)
from unilab.base.contracts.observations import get_obs_dims, split_obs_dict
from unilab.base.contracts.probe import EnvContractProbe, probe_env_contract

__all__ = [
    "EnvAlgoCapabilities",
    "EnvAlgoCapabilitiesProtocol",
    "EnvContractProbe",
    "EnvFactory",
    "EnvPlayCapabilitiesProtocol",
    "EnvProtocol",
    "EnvStateProtocol",
    "SupportsAlgoCapabilitiesProtocol",
    "SupportsBackendDiagnosticsProtocol",
    "SupportsCompletionReceiptsProtocol",
    "get_algo_capabilities",
    "get_backend_runtime_diagnostics",
    "get_obs_dims",
    "get_produces_completion_receipts",
    "probe_env_contract",
    "split_obs_dict",
]
