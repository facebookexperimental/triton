"""Compatibility entry point for the gfx950 inter-wave GEMM tutorial."""

from triton.tlx.ops.kernels.mm.gfx950 import (
    _iw_KERNEL_NAME as KERNEL_NAME,
    _iw_MIN_K as MIN_K,
    _iw_matmul as matmul,
    _iw_streamk_matmul as streamk_matmul,
)

__all__ = ["KERNEL_NAME", "MIN_K", "matmul", "streamk_matmul"]
