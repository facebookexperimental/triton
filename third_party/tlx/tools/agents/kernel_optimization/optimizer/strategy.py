_LOCAL_SEARCH_GUIDANCE = """After a measured gain, consider an untried, source-derived legal neighbor when evidence
and remaining budget justify it. Isolate one hypothesis and preserve numerical, masking,
barrier, and lifetime invariants. Interpret a failed value narrowly: it does not disprove
an entire hypothesis family. Treat noisy or inconclusive measurements as inconclusive.
Balance local refinement with other supported hypotheses; do not mandate a sweep or a
parameter sequence."""

CANDIDATE_ONLY_STRATEGY = (
    """1. Begin with measured evidence and feedback from the decision maker.
2. Try low-effort scopes first: a config change or kernel change.
"""
    + _LOCAL_SEARCH_GUIDANCE
)

OPTIMIZATION_STRATEGY = (
    """1. Begin with measured evidence and feedback from the decision maker.
2. Try low-effort scopes first: a config change or kernel change.
3. Escalate when evidence supports it. Modify PTX or AMDGCN experimentally and override IR. Convert successful signals into a well-scoped compiler change.
4. Escalate broader compiler changes to a human.
"""
    + _LOCAL_SEARCH_GUIDANCE
)

__all__ = ["CANDIDATE_ONLY_STRATEGY", "OPTIMIZATION_STRATEGY"]
