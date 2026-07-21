"""Benchmark spec for case3 (Flash-Attention fwd) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors perf_generated_vs_handwritten.py (gen + hw side by side).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

BLOCK_M, BLOCK_N, HEAD_DIM = 128, 64, 128
TOL = 1e-2

SHAPES = [
    (1, 4, 512),
    (1, 8, 1024),
    (2, 16, 2048),
    (1, 16, 4096),
    (2, 16, 4096),
    (1, 32, 8192),
]


def make_inputs(shape):
    Z, H, N_CTX = shape
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    m_lse = torch.empty(Z * H, N_CTX, device="cuda", dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    return {
        "q": q, "k": k, "v": v, "out": out, "m_lse": m_lse, "sm": sm_scale,
        "Z": Z, "H": H, "N_CTX": N_CTX,
        "qf": q.contiguous().view(-1, HEAD_DIM),
        "kf": k.contiguous().view(-1, HEAD_DIM),
        "vf": v.contiguous().view(-1, HEAD_DIM),
        "of": out.view(-1, HEAD_DIM),
    }


def _grid(inputs):
    return (triton.cdiv(inputs["N_CTX"], BLOCK_M), inputs["Z"] * inputs["H"])


def gen_call(generated, inputs):
    generated.fa_fwd_kernel_nows[_grid(inputs)](
        inputs["qf"], inputs["kf"], inputs["vf"], inputs["of"], inputs["m_lse"], inputs["sm"],
        inputs["Z"] * inputs["H"], inputs["N_CTX"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["out"]


def hw_call(handwritten, inputs):
    handwritten.fa_fwd_kernel[_grid(inputs)](
        inputs["qf"], inputs["kf"], inputs["vf"], inputs["of"], inputs["m_lse"], inputs["sm"],
        inputs["Z"], inputs["H"], inputs["N_CTX"],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
    )
    return inputs["out"]


def metric(shape):
    Z, H, N_CTX = shape
    return (4 * Z * H * N_CTX * N_CTX * HEAD_DIM, 1e12, "TFLOPS")


def reference(inputs):
    return F.scaled_dot_product_attention(inputs["q"], inputs["k"], inputs["v"], scale=inputs["sm"])
