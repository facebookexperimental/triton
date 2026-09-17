"""Shared L1/L2 shapes for the gfx950 TorchTLX ``addmm`` provider."""

from __future__ import annotations

from ..mm._shapes import operand

# Entries are [M, N, K, a_strides, b_strides, dtype]. The bias is a dense
# length-N vector. A is (M, K), B is (K, N); strides preserve row/column major.
SYNTHETIC: list[list] = [
    [256, 256, 272, (272, 1), (1, 272), "fp16"],
    [256, 256, 272, (272, 1), (1, 272), "bf16"],
    [264, 256, 328, (328, 1), (1, 328), "fp16"],
    # Unaligned K selects the register-load fallback rather than async LDS.
    [256, 192, 259, (259, 1), (1, 259), "fp16"],
]

# Representative MI350X regimes already covered by the lower-level template
# tests: thin-N warp pipeline plus inter-wave and large-M cases. The odd-K
# register fallback stays in L1; it is not yet competitive enough for the L2
# speed gate.
GFX950_FOCUS: list[list] = [
    [4096, 192, 2048, (2048, 1), (1, 2048), "fp16"],
    [4096, 192, 2048, (2048, 1), (1, 2048), "bf16"],
    [1024, 1024, 864, (864, 1), (1, 864), "fp16"],
    [64000, 256, 256, (256, 1), (1, 256), "fp16"],
]


def inputs(entry, dtype, device="cuda"):
    """Construct the bias, A, and B tensors described by one shape entry."""
    m, n, k, a_strides, b_strides, _ = entry
    import torch

    bias = torch.randn(n, device=device, dtype=dtype)
    a = operand(m, k, a_strides, dtype, device=device)
    b = operand(k, n, b_strides, dtype, device=device)
    return bias, a, b


def flops(m, n, k):
    return 2 * m * n * k


def label(m, n, k, a_strides, b_strides, dtype) -> str:
    strides = f"[[{a_strides[0]}, {a_strides[1]}], [{b_strides[0]}, {b_strides[1]}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{k}', 'bias': '[{n}]'}})")
