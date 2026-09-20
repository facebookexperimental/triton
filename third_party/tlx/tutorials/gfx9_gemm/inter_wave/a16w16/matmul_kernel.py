"""Compatibility entry point for the gfx950 inter-wave GEMM tutorial."""

from triton.tlx.ops.kernels.mm.gfx950 import (
    KERNEL_NAME,
    MIN_K,
    _lds_matmul as matmul,
    streamk_matmul,
)

__all__ = ["KERNEL_NAME", "MIN_K", "matmul", "streamk_matmul"]
