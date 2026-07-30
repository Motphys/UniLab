"""Domain randomization package.

Invariant: this package must not depend on unilab.base.*
"""

from .keyed_rng import (
    KEYED_RNG_ALGORITHM,
    KeyedRandomBatch,
    KeyedRandomContractError,
    KeyedRandomSpec,
    KeyedRandomStream,
    KeyedRandomTrafficDiagnostics,
    RandomCorrelation,
    RandomDistribution,
    StaleKeyedRandomBatchError,
    keyed_random_reference,
)
from .manager import DomainRandomizationManager
from .provider import DomainRandomizationProvider
from .types import (
    DomainRandomizationCapabilities,
    DomainRandomizationExecutionMode,
    GeomSizeOverride,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ModelVariantSpec,
    ResetPlan,
    ResetRandomizationPayload,
    UnsupportedDomainRandomizationError,
)

__all__ = [
    "KEYED_RNG_ALGORITHM",
    "DomainRandomizationCapabilities",
    "DomainRandomizationExecutionMode",
    "DomainRandomizationManager",
    "DomainRandomizationProvider",
    "GeomSizeOverride",
    "InitRandomizationPlan",
    "IntervalRandomizationPlan",
    "KeyedRandomBatch",
    "KeyedRandomContractError",
    "KeyedRandomSpec",
    "KeyedRandomStream",
    "KeyedRandomTrafficDiagnostics",
    "ModelVariantSpec",
    "ResetPlan",
    "ResetRandomizationPayload",
    "RandomCorrelation",
    "RandomDistribution",
    "StaleKeyedRandomBatchError",
    "UnsupportedDomainRandomizationError",
    "keyed_random_reference",
]
