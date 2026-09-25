"""MI300X (gfx942/CDNA3) ``tlx.ops.addmm`` implementation."""

from ..mm.gfx942 import _gemm


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Compute ``input + a @ b`` with the shared gfx942 GEMM kernel.

    ``input`` may be ``(N,)`` or a two-dimensional tensor broadcastable to
    ``(M, N)``. Matrix and input scale factors are both one.
    """
    return _gemm(a, b, input, out=out, space=space)
