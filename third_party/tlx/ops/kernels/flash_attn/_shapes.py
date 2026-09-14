from __future__ import annotations

# Entries are [Z, H, N_CTX, HEAD_DIM, causal, dtype].

#: Identical to `test_flash_attn.py::SHAPES`.
SYNTHETIC: list[list] = [
    [1, 1, 256, 64, False, "fp16"],
    [1, 2, 512, 64, True, "fp16"],
    [2, 4, 1024, 64, False, "fp16"],
    [2, 4, 1024, 128, False, "fp16"],
    [2, 4, 1024, 128, True, "fp16"],
    [4, 8, 2048, 128, True, "fp16"],
    [1, 16, 4096, 128, False, "fp16"],
    [2, 32, 2048, 64, False, "fp16"],
    [4, 8, 512, 64, True, "fp16"],
    [1, 1, 8192, 128, True, "fp16"],
]

#: TODO: placeholder shapes, not a capture. Awaiting real ones.
SM100_FOCUS: list[list] = [
    [4, 32, 4096, 128, False, "bf16"],
    [4, 32, 4096, 128, True, "bf16"],
    [2, 32, 8192, 128, True, "bf16"],
    [1, 16, 16384, 128, True, "bf16"],
    [4, 32, 4096, 64, False, "bf16"],
    [4, 32, 4096, 128, False, "fp16"],
]


def qkv(Z, H, N_CTX, HEAD_DIM, dtype, requires_grad=False, device="cuda"):
    import torch

    return [
        torch.randn((Z, H, N_CTX, HEAD_DIM), device=device, dtype=dtype).requires_grad_(requires_grad) for _ in range(3)
    ]


def flops(Z, H, N_CTX, HEAD_DIM, causal, direction="fwd"):
    """The tutorials' and tritonbench's count, so the numbers are comparable.

    `tutorials/fused_attention_ws_auto_tma.py`: 2.5x on the backward is 2.0 plus
    0.5 to recompute the scores.
    """
    total = 2 * (2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM)
    if causal:
        total *= 0.5
    if direction == "bwd":
        total *= 2.5
    return int(total)


def label(Z, H, N_CTX, HEAD_DIM, causal, dtype, direction="fwd") -> str:
    """The report's input column."""
    return (f"((), {{'dtype': '{dtype}', 'causal': '{causal}', 'dir': '{direction}', "
            f"'Z': '{Z}', 'H': '{H}', 'N_CTX': '{N_CTX}', 'HEAD_DIM': '{HEAD_DIM}'}})")
