"""The two exceptions the package raises.

Their own module so that every other module can import them without pulling in ctypes, torch
or triton: `ltapi` raises `CublasUnsupportedShape` when cuBLAS has no algo for a shape, `arch`
raises `CublasUnsupportedPlatform` for a GPU nobody has measured, and `gemm` catches both.
"""
from __future__ import annotations


class CublasUnsupportedShape(Exception):
    """No Triton reconstruction bit-matches cuBLAS for this shape, even with a runtime
    byte-compare (e.g. fp8 vertical/cluster split-K, fp16 non-aligned s1688 K=8 / odd-K)."""


class CublasUnsupportedPlatform(CublasUnsupportedShape):
    """No measured strategy for this GPU architecture.

    Every reconstruction rule in this file was measured on a GPU, not derived, so it must not be
    extrapolated to one we have not run on -- even though sm_100 and sm_103 did turn out to
    agree. Subclasses `CublasUnsupportedShape` on purpose, so a caller that already writes
    `except CublasUnsupportedShape: <fall back to cuBLAS>` keeps working unchanged on a machine
    we have not measured."""
