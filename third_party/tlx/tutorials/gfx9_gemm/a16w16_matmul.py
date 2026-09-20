"""Compatibility entry point for the gfx950 ``tlx.ops.mm`` implementation."""

from triton.tlx.ops import mm as _ops_mm


def matmul(a, b):
    """Dispatch through the production gfx950 MM implementation."""
    return _ops_mm(a, b, arch="gfx950")
