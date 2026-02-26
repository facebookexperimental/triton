"""Rule engine for kernel heuristics.

Two independent parsers for two YAML formats:

- ``RuleEngine``      — ordered first-match rules (the knowledge base)
- ``CandidateScorer`` — scored fallback search  (the safety net)
"""

from .candidates import CandidateScorer
from .engine import RuleEngine

__all__ = ["RuleEngine", "CandidateScorer"]
