"""Family-level gfx950 a16w16 GEMM dispatcher.

This module intentionally sits above the implementation directories: it
selects between an exact-shape persistent kernel and the general inter-wave
fallback.  Keeping dispatch policy here avoids making either implementation
depend on the other and gives clients one stable a16w16 entry point.
"""

from .a16w16.matmul_kernel_persistent import (
    matmul as _persistent_matmul,
    supports as _persistent_supports,
)
from .inter_wave.a16w16.matmul_kernel import matmul as _inter_wave_matmul


def matmul(a, b):
    """Dispatch to a specialized persistent kernel or the general fallback."""
    if _persistent_supports(a, b):
        return _persistent_matmul(a, b)
    return _inter_wave_matmul(a, b)
