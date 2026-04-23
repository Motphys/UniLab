from __future__ import annotations

import warnings

from unilab.base.final_observation import (
    TerminalObservationContract,
    TransitionBootstrapContract,
    patch_transition_next_obs,
    resolve_terminal_observation_contract,
    resolve_transition_bootstrap_contract,
)

warnings.warn(
    "`unilab.utils.final_observation` is deprecated and will be removed in 0.2.0; "
    "use `unilab.base.final_observation` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TerminalObservationContract",
    "TransitionBootstrapContract",
    "patch_transition_next_obs",
    "resolve_terminal_observation_contract",
    "resolve_transition_bootstrap_contract",
]
