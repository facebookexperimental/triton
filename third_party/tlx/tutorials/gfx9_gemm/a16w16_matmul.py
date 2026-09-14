"""Family-level gfx950 a16w16 GEMM dispatcher.

This module intentionally sits above the implementation directories: it
selects an exact-shape persistent kernel, a skinny LocalSplitU kernel, or the
general inter-wave fallback. Keeping dispatch policy here avoids making the
implementations depend on one another and gives clients one stable a16w16
entry point.
"""

from .a16w16.matmul_kernel_persistent import (
    matmul as _persistent_matmul,
    supports as _persistent_supports,
)
from .inter_wave.a16w16.matmul_kernel import matmul as _inter_wave_matmul
# This in-tree compatibility dispatcher intentionally uses the implementation
# fast path. Routing every ~12-us launch back through the public catalog adds
# measurable Python dispatch overhead; external callers should use tlx.ops.mm.
from triton.tlx.ops.kernels.mm.gfx950 import (
    matmul as _skinny_matmul,
    supports as _skinny_supports,
)


def matmul(a, b):
    """Dispatch to a specialized kernel or the general fallback."""
    if _persistent_supports(a, b):
        return _persistent_matmul(a, b)
    if _skinny_supports(a, b):
        return _skinny_matmul(a, b)
    return _inter_wave_matmul(a, b)
