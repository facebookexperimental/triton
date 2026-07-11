"""Benchmark spec for case5 (persistent addmm + 2D bias) consumed by
examples/testing/perf_engine.py.

Launch logic mirrors run_generated.py (generated) and run_handwritten.py
(reference). Both kernels take TMA tensor descriptors and share the output
buffer; perf_engine re-runs the generated kernel after the handwritten call
so the timed outputs are not clobbered.
"""

from __future__ import annotations

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
NUM_BUFFERS_AB, NUM_BUFFERS_ACC = 2, 2
TOL = 5e-3

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]


def make_inputs(shape):
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    return {
        "a": a,
        "b": b,
        "bias": bias,
        "c": c,
        "a_desc": TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K]),
        "b_desc": TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N]),
        "bias_desc": TensorDescriptor.from_tensor(bias, [BLOCK_M, BLOCK_N]),
        "c_desc": TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N]),
        "M": M,
        "N": N,
        "K": K,
        "NUM_SMS": torch.cuda.get_device_properties(0).multi_processor_count,
    }


def gen_call(generated, inputs):
    grid = (inputs["NUM_SMS"],)
    generated.addmm_persistent_2d_bias[grid](
        inputs["a_desc"], inputs["b_desc"], inputs["bias_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c"]


def hw_call(handwritten, inputs):
    grid = (inputs["NUM_SMS"],)
    handwritten.addmm_persistent_2d_bias[grid](
        inputs["a_desc"], inputs["b_desc"], inputs["bias_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        NUM_SMS=inputs["NUM_SMS"],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_BUFFERS_AB=NUM_BUFFERS_AB, NUM_BUFFERS_ACC=NUM_BUFFERS_ACC,
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c"]


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    return (
        torch.matmul(inputs["a"].float(), inputs["b"].float()) + inputs["bias"].float()
    ).to(torch.float16)
