"""SimBackend public-method contract inventory (Phase 6 safety net; locks current contract).

Pins the *name sets* of the three method categories of ``SimBackend``:
- ``abstract``  : ``@abc.abstractmethod`` (required — subclasses must implement)
- ``optional``  : concrete body that ``raise NotImplementedError`` (subclass may override)
- ``concrete``  : real default implementation

b.md b4: lock the NAME SETS, not just counts — a "delete one required abstract + mis-promote
one optional" swap leaves the count unchanged but breaks the contract. When Phase 6 splits
SimBackend into 6 Mixin ABCs, the composed class must still classify identically, so the
abstract→optional demotion the plan warns against is caught here.

Baseline: ``simbackend_contract_baseline.json`` (currently 28 / 29 / 13). Regenerate only
on an intentional contract change (see ``_dump_baseline`` below).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from unilab.base.backend import base

BASELINE = Path(__file__).resolve().parent / "simbackend_contract_baseline.json"


def classify_public_methods(cls: type) -> dict[str, list[str]]:
    """Classify every public attribute of ``cls`` as abstract / optional / concrete.

    Uses ``getattr_static`` (never triggers descriptors) + ``__isabstractmethod__`` +
    source inspection for ``raise NotImplementedError``. Deterministic, so generator and
    test agree.
    """
    out: dict[str, list[str]] = {"abstract": [], "optional": [], "concrete": []}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        if getattr(attr, "__isabstractmethod__", False):
            out["abstract"].append(name)
            continue
        target = attr
        if isinstance(attr, property):
            target = attr.fget
        elif isinstance(attr, (staticmethod, classmethod)):
            target = attr.__func__
        try:
            src = inspect.getsource(target) if target is not None else ""
        except (TypeError, OSError):
            src = ""
        bucket = "optional" if "raise NotImplementedError" in src else "concrete"
        out[bucket].append(name)
    return {key: sorted(values) for key, values in out.items()}


def _dump_baseline() -> None:
    """Regenerate the baseline after an intentional contract change. Run manually::

    uv run python -c "from tests.base.test_simbackend_contract import _dump_baseline; _dump_baseline()"
    """
    BASELINE.write_text(json.dumps(classify_public_methods(base.SimBackend), indent=2) + "\n")


def test_simbackend_contract_unchanged():
    expected = json.loads(BASELINE.read_text())
    current = classify_public_methods(base.SimBackend)
    if current == expected:
        return
    drift = []
    for category in ("abstract", "optional", "concrete"):
        added = sorted(set(current[category]) - set(expected[category]))
        removed = sorted(set(expected[category]) - set(current[category]))
        if added or removed:
            drift.append(f"  {category}: added={added} removed={removed}")
    raise AssertionError(
        "SimBackend public-method contract drifted from baseline:\n"
        + "\n".join(drift)
        + "\n(If intentional, regenerate via _dump_baseline; do not weaken required abstracts.)"
    )
