"""GEMM shape parsing for the torchTLX perf harness.

Shapes are (M, N, K). Batch runs take an explicit ``--shapes MxNxK,...`` list.
"""


def parse_shapes(spec: str):
    """Parse a ``MxNxK,MxNxK,...`` string into a list of (M, N, K) tuples."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.lower().replace(" ", "").split("x")
        if len(parts) != 3:
            raise ValueError(f"bad shape {chunk!r}; expected MxNxK")
        out.append(tuple(int(p) for p in parts))
    return out
