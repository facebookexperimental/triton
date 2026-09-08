# Optimizer

The Optimizer is the only autonomous, Codex-backed component. It consumes measured
profiling evidence and Decision Maker feedback, forms one testable hypothesis, and returns
a candidate plus rationale. It cannot declare correctness, performance, or promotion.

Candidate generation, source handling, and target guidance live here. The Decision Maker
owns execution and every authoritative verdict.

`strategy.py` contains the fixed evidence-first escalation policy. `knowledge/` separates
common guidance from vendor- and architecture-specific guidance selected by a registry,
so adding a GPU architecture does not add conditionals to the Codex adapter.
