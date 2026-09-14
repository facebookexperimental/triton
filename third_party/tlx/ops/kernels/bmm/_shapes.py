"""Shared L1/L2 shapes for the gfx950 TorchTLX ``bmm`` provider."""

from __future__ import annotations

# Entries are [B, M, N, K, a_strides, b_strides, dtype]. A is (B, M, K),
# B is (B, K, N). A batch stride of zero selects the shared-LHS specialization.
SYNTHETIC: list[list] = [
    [8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "fp16"],
    [8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "bf16"],
    # Unaligned K selects the register-load fallback rather than async LDS.
    [2, 128, 128, 259, (128 * 259, 259, 1), (259 * 128, 128, 1), "fp16"],
    # Exact shape handled by the shared-LHS specialization.
    [2, 40, 256, 1956, (0, 1956, 1), (1956 * 256, 256, 1), "fp16"],
]

# The general aligned path and the specialized shared-LHS path are the current
# performance targets. The odd-K register fallback remains L1-only.
GFX950_FOCUS: list[list] = [
    [8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "fp16"],
    [8, 256, 256, 272, (256 * 272, 272, 1), (272 * 256, 256, 1), "bf16"],
    [2, 448, 160, 931, (0, 931, 1), (931 * 160, 160, 1), "fp16"],
    [2, 1195, 256, 2309, (0, 2309, 1), (2309 * 256, 256, 1), "fp16"],
]


def operand(batch, rows, cols, strides, dtype, device="cuda"):
    """A dense or shared-batch tensor with exactly the recorded strides."""
    import torch

    batch_stride, row_stride, col_stride = strides
    if row_stride != cols or col_stride != 1:
        raise ValueError(f"unsupported bmm strides {strides}")
    if batch_stride == 0:
        return torch.randn((rows, cols), device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    if batch_stride == rows * cols:
        return torch.randn((batch, rows, cols), device=device, dtype=dtype)
    raise ValueError(f"unsupported bmm strides {strides}")


def inputs(entry, dtype, device="cuda"):
    batch, m, n, k, a_strides, b_strides, _ = entry
    a = operand(batch, m, k, a_strides, dtype, device=device)
    b = operand(batch, k, n, b_strides, dtype, device=device)
    return a, b


def flops(batch, m, n, k):
    return 2 * batch * m * n * k


def label(batch, m, n, k, a_strides, b_strides, dtype) -> str:
    strides = f"[[{', '.join(map(str, a_strides))}], [{', '.join(map(str, b_strides))}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', 'B': '{batch}', "
            f"'M': '{m}', 'N': '{n}', 'K': '{k}'}})")
