"""Standalone fused addmm for gfx950 using the inter-wave TLX GEMM pipeline.

The operation is ``out = bias + a @ b``. ``a`` must be row-major and ``b``
must be column-major so the kernel can issue coalesced direct-to-LDS loads.
The bias may be one-dimensional ``(N,)`` or broadcastable to ``(M, N)``.
"""

import torch

from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    BLOCK_K,
    _launch,
)


def addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, SPLIT_K=None):
    """Return ``bias + a @ b`` using the gfx950 TLX kernel."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("addmm expects two-dimensional matrix operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if not a.is_contiguous():
        raise ValueError("a must be row-major contiguous")
    if b.stride(0) != 1:
        raise ValueError("b must be column-major (stride(0) == 1)")

    M, K = a.shape
    _, N = b.shape
    if K < 2 * BLOCK_K or K % BLOCK_K != 0:
        raise ValueError(f"K={K} must be at least {2 * BLOCK_K} and a multiple of {BLOCK_K}")
    try:
        bias_2d = torch.broadcast_to(bias, (M, N))
    except RuntimeError as error:
        raise ValueError(f"Bias shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})") from error
    return _launch(a, b, bias=bias_2d, SPLIT_K=SPLIT_K)
