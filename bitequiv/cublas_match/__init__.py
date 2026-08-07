"""cublas-match: a Triton GEMM whose output is bit-identical to cuBLAS.

    from bitequiv.cublas_match import cublas_equivalent_gemm, cublas_matmul

Query cuBLAS's own heuristic, read the recipe off the algo config it returns, and run the
Triton kernel that reproduces that arithmetic.  See README.md.
"""
from __future__ import annotations

from .errors import CublasUnsupportedPlatform, CublasUnsupportedShape
from .gemm import (cublas_equivalent_gemm, cublas_equivalent_scaled_mm, plan_origin, verify)
from .ltapi import cublas_matmul, cublaslt_version, set_cublaslt, set_workspace_bytes
from .plan import CublasGemmPlan, static_plan

__all__ = [
    "CublasGemmPlan",
    "CublasUnsupportedPlatform",
    "CublasUnsupportedShape",
    "cublas_equivalent_gemm",
    "cublas_equivalent_scaled_mm",
    "cublas_matmul",
    "cublaslt_version",
    "plan_origin",
    "set_cublaslt",
    "set_workspace_bytes",
    "static_plan",
    "verify",
]
