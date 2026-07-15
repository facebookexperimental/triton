"""Benchmark spec for case5 (persistent addmm + 2D bias) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (generated) and run_handwritten.py
(hand-written TLX-WS reference). The generated kernel is a persistent GEMM that
bakes the B200 SM count into its tile schedule, so the launch grid is the device
multiprocessor count.
"""

from __future__ import annotations

import torch
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
NUM_BUFFERS_AB, NUM_BUFFERS_ACC = 2, 2
TOL = 5e-3

SHAPES = [
    (256, 256, 128),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1024, 1024, 16384),
]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(shape):
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    # Separate output buffers so the generated and hand-written results stay
    # comparable (gen writes c, hw writes c_hw); a/b/bias are read-only.
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    c_hw = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    return {
        "a": a, "b": b, "bias": bias, "c": c, "c_hw": c_hw,
        "M": M, "N": N, "K": K,
        "a_desc": TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K]),
        "b_desc": TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N]),
        "bias_desc": TensorDescriptor.from_tensor(bias, [BLOCK_M, BLOCK_N]),
        "c_desc": TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N]),
        "c_hw_desc": TensorDescriptor.from_tensor(c_hw, [BLOCK_M, BLOCK_N]),
    }


def gen_call(generated, inputs):
    generated.addmm_persistent_2d_bias[(_num_sms(),)](
        inputs["a_desc"], inputs["b_desc"], inputs["bias_desc"], inputs["c_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c"]


def hw_call(handwritten, inputs):
    nsms = _num_sms()
    handwritten.addmm_persistent_2d_bias[(nsms,)](
        inputs["a_desc"], inputs["b_desc"], inputs["bias_desc"], inputs["c_hw_desc"],
        inputs["M"], inputs["N"], inputs["K"],
        NUM_SMS=nsms, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_BUFFERS_AB=NUM_BUFFERS_AB, NUM_BUFFERS_ACC=NUM_BUFFERS_ACC,
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c_hw"]


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")  # bias add is negligible vs the GEMM


def reference(inputs):
    return (torch.matmul(inputs["a"].float(), inputs["b"].float()) + inputs["bias"].float()).to(torch.float16)
