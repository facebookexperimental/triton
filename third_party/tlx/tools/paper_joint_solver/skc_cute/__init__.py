"""Retired FA4 parameter-shim experiment.

This package remains only to fail old callers clearly.  It is not a compiler
backend.  The supported path emits Twill IR for an expert manual CUDA C++
implementation via ``skc.compiler.prepare_manual_cuda_handoff``.
"""

from .binder_cute import InheritedFA4BackendRemoved

__all__ = ["InheritedFA4BackendRemoved"]
