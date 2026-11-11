from __future__ import annotations

# Muse-Spark-Open-816B-A42B at TP8 all-reduces bf16 [tokens, 8192] activations after attention and MoE.
MUSE_HIDDEN = 8192

# Element counts on both sides of the 8-rank 80 KB one-stage/two-stage switch: decode (1-4 tokens), the DFLASH
# verify batch (8 draft tokens + 1), prefill chunks, and lengths that do not divide the rank count.
CORRECTNESS_NUMELS = tuple(MUSE_HIDDEN * t for t in (1, 2, 4, 5, 9, 16, 36, 72, 256)) + (1000, 8 * 1000 + 3)

# Decode-style messages exercised under HIP-graph capture.
GRAPH_TOKENS = (4, 9)


def label(numel: int) -> str:
    return f"numel={numel} ({numel * 2} B bf16)"
