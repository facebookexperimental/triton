"""Benchmark spec for case1 (simple GEMM) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (generated) and handwritten.gemm (reference).
"""

from __future__ import annotations

import torch
import triton

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
TOL = 5e-3

SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]


def make_inputs(shape):
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    return {"a": a, "b": b, "c": c, "M": M, "N": N, "K": K}


def gen_call(generated, inputs):
    a, b, c = inputs["a"], inputs["b"], inputs["c"]
    M, N, K = inputs["M"], inputs["N"], inputs["K"]
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    generated.gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return c


def hw_call(handwritten, inputs):
    return handwritten.gemm(inputs["a"], inputs["b"])


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    return torch.matmul(inputs["a"], inputs["b"])
