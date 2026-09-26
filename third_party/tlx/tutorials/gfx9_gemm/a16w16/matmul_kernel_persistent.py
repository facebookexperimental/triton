"""Compatibility entry point for the gfx950 persistent GEMM tutorial."""

from triton.tlx.ops.kernels.mm.gfx950 import (
    _launch_persistent as matmul,
    _persistent_supports as supports,
)

__all__ = ["matmul", "supports"]
