OPTIMIZATION_STRATEGY = """1. Begin with measured evidence and feedback from the decision maker.
2. Try low-effort scopes first: a config change or kernel change.
3. Escalate when evidence supports it. Modify PTX or AMDGCN experimentally and override IR. Convert successful signals into a well-scoped compiler change.
4. Escalate broader compiler changes to a human."""

__all__ = ["OPTIMIZATION_STRATEGY"]
