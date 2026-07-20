"""Benchmark spec for case9 blockwise fp8 scaled_mm, consumed by perf_harness.py.

Blockwise (DeepSeek) fp8 e4m3 scaled_mm: D = (A @ B^T) * (sa[m,g] * sb[nblk,g])
summed per 128-K group. Launch mirrors run_generated.py; reference mirrors the
torch blockwise _ref. The hand-written WS reference is the autotuned
blackwell_scaled_mm_ws wrapper (mirrors perf_generated_vs_handwritten.py).
"""

from __future__ import annotations

import torch
import triton

NUM_SMS = 148
TOL = 5e-2  # fp8 tolerance

SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def make_inputs(shape):
    M, N, K = shape
    a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    # scale_a M-major (outer-dim-major): stride (1, M); scale_b row-major.
    scale_a = (
        torch.rand(M, K // 128, device="cuda", dtype=torch.float32).t().contiguous().t()
    )
    scale_b = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    return {"a": a, "b": b, "scale_a": scale_a, "scale_b": scale_b, "c": c,
            "M": M, "N": N, "K": K}


def gen_call(generated, inputs):
    a, b = inputs["a"], inputs["b"]
    scale_a, scale_b, c = inputs["scale_a"], inputs["scale_b"], inputs["c"]
    M, N, K = inputs["M"], inputs["N"], inputs["K"]
    grid = (NUM_SMS,)
    generated._scaled_mm_blockwise[grid](
        a, b, c, scale_a, scale_b, M, N, K,
        a.stride(0),        # stride_am (= K)
        b.stride(0),        # stride_bn (= K)
        c.stride(0),        # stride_cm (= N)
        scale_a.stride(1),  # stride_sa_g (= M, M-major)
        scale_b.stride(0),  # stride_sb_n (= K//128)
        num_warps=8, num_ctas=1, num_stages=2,
    )
    return c


def hw_call(handwritten, inputs):
    return handwritten.blackwell_scaled_mm_ws(
        inputs["a"], inputs["b"], inputs["scale_a"], inputs["scale_b"],
        scale_mode="blockwise",
    )


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    a, b = inputs["a"], inputs["b"]
    scale_a, scale_b = inputs["scale_a"], inputs["scale_b"]
    M, K = a.shape
    N = b.shape[0]
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    for g in range(K // 128):
        ak = af[:, g * 128 : (g + 1) * 128]
        bk = bf[:, g * 128 : (g + 1) * 128]
        partial = ak @ bk.t()
        sa = scale_a[:, g][:, None]
        sb = scale_b[:, g].repeat_interleave(128)[None, :]
        out += partial * sa * sb
    return out.to(torch.bfloat16)
