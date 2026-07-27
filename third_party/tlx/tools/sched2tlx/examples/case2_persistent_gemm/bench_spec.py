"""Benchmark spec for case2 (persistent GEMM) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (generated) and run_handwritten.py (reference).
Large shapes are included so the persistent outer loop runs many tiles — the exact
regime where the pre-fix emitter's name collision corrupts tile addressing.
"""

from __future__ import annotations

import torch
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
TOL = 5e-3

SHAPES = [
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    # Tiny-K (K=256 -> k_tiles=4) with a SMEM ring deeper than k_tiles used to
    # wedge the hand-written kernel (depths 5/6 hung; 2/3/4 passed). Root cause
    # was the per-tile TMEM toggle desyncing against the continuous cross-tile
    # SMEM ring; the handshake now uses a continuous TMEM counter like the
    # tutorial, so any ring depth is safe at any K.
    (8192, 8192, 256),
]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(shape):
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    # The hand-written matmul_kernel loads via TMA descriptors; b is laid out as
    # [N, K] so TMA loads [offs_bn, offs_k].
    b_t = b.t().contiguous()
    return {
        "a": a, "b": b, "c": c, "M": M, "N": N, "K": K,
        "a_desc": TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K]),
        "b_desc": TensorDescriptor.from_tensor(b_t, [BLOCK_N, BLOCK_K]),
        "c_desc": TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N]),
    }


def gen_call(generated, inputs):
    # Generated kernel is a raw-pointer persistent GEMM (not autotuned) — keep
    # num_warps/num_ctas/num_stages, mirroring run_generated.py.
    a, b, c = inputs["a"], inputs["b"], inputs["c"]
    generated._gemm_persistent[(_num_sms(),)](
        a, b, c, inputs["M"], inputs["N"], inputs["K"],
        a.stride(0), b.stride(0), c.stride(0),
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return c


def hw_call(handwritten, inputs):
    # Hand-written kernel is the descriptor-based, autotuned matmul_kernel;
    # buffer depths + num_warps/num_ctas/num_stages come from @triton.autotune.
    nsms = _num_sms()
    handwritten.matmul_kernel[(nsms,)](
        inputs["a_desc"], inputs["b_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        NUM_SMS=nsms, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return inputs["c"]


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    return torch.matmul(inputs["a"], inputs["b"])
