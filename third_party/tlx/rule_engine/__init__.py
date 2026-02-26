"""Rule engine for kernel heuristics.

Two independent parsers for two YAML formats:

- ``RuleEngine``         — ordered first-match rules (the knowledge base)
- ``CandidateScorer``    — scored fallback search  (the safety net)

Standalone validation for use by other parsers:

- ``validate_rules_yaml`` — validate a rules YAML without building a RuleEngine
- ``validate_expr``       — validate a single expression string
"""

from .candidates import CandidateScorer
from .engine import RuleEngine, validate_rules_yaml
from .expr import validate_expr

__all__ = ["RuleEngine", "CandidateScorer", "validate_rules_yaml", "validate_expr"]
