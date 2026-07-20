"""Benchmark spec for case2 (persistent GEMM) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (generated) and run_handwritten.py (reference).
Large shapes are included so the persistent outer loop runs many tiles — the exact
regime where the pre-fix emitter's name collision corrupts tile addressing.
"""

from __future__ import annotations

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
NUM_SMEM_BUFFERS = 2
TOL = 5e-3

SHAPES = [
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (8192, 8192, 256),
]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(shape):
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_t = b.t().contiguous()  # [N, K] so TMA loads [offs_bn, offs_k]
    b_desc = TensorDescriptor.from_tensor(b_t, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
    return {"a": a, "b": b, "c": c, "a_desc": a_desc, "b_desc": b_desc, "c_desc": c_desc,
            "M": M, "N": N, "K": K}


def gen_call(generated, inputs):
    grid = (_num_sms(),)
    generated.matmul_kernel_tma_persistent_simple[grid](
        inputs["a_desc"], inputs["b_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c"]


def hw_call(handwritten, inputs):
    nsms = _num_sms()
    handwritten.matmul_kernel[(nsms,)](
        inputs["a_desc"], inputs["b_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        NUM_SMS=nsms, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS, num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c"]


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    return torch.matmul(inputs["a"], inputs["b"])
