from __future__ import annotations

# Entries are [Z, H, N_CTX, HEAD_DIM, causal, dtype].
SM100_FOCUS: list[list] = [
    [4, 32, 1024, 128, False, "bf16"],
    [4, 32, 1024, 128, True, "bf16"],
    [4, 32, 2048, 128, False, "bf16"],
    [4, 32, 2048, 128, True, "bf16"],
    [4, 32, 4096, 128, False, "bf16"],
    [4, 32, 4096, 128, True, "bf16"],
    [4, 32, 8192, 128, False, "bf16"],
    [4, 32, 8192, 128, True, "bf16"],
]
