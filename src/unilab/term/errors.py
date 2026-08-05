"""Stable diagnostics for structured term registration and execution."""


class TermContractError(ValueError):
    """Base class for invalid term definitions, plans, or bound views."""


class TermRegistrationError(TermContractError):
    """Raised when a registry entry violates the definition contract."""


class TermPlanError(TermContractError):
    """Raised when configured terms cannot form one deterministic plan."""


class TermBindingError(TermContractError):
    """Raised when runtime arrays do not satisfy a resolved plan."""
