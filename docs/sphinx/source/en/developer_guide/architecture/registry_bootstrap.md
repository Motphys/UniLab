# Registry Bootstrap

UniLab uses an explicit registration phase at process start: every task,
backend, and algorithm registers itself by name into a global registry.
The CLI then resolves `--algo / --task / --sim` triples via the registry.

ADR-0004 ({doc}`../adr/ADR-0004-registry-bootstrap-contract`) defines the
bootstrap contract: registration ordering, error semantics on
duplicates, and the rule that **no I/O happens during registration**.

See {py:mod}`unilab.base.registry` for the implementation.
