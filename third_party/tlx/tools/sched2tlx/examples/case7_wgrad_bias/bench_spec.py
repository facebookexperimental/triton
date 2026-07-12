"""Benchmark spec for case7 (fused wgrad GEMM + bias-gradient reduce) consumed by examples/testing/perf_regression/perf_harness.py.

Computes the linear-layer backward gradients dW = doutᵀ @ act and db = dout.sum(0).
Launch logic mirrors run_generated.py (generated `wgrad_bias_nows`) and
run_handwritten.py (hand-written `wgrad_bias_ws`). The harness compares the
dominant GEMM output dW; db is produced by both kernels during timing but, like
the other cases' single-output contract, is not separately asserted here. The
persistent kernels bake the B200 SM count into their tile schedule, so the launch
grid is the device multiprocessor count.
"""

from __future__ import annotations

import torch

BLOCK_KO, BLOCK_NI, BLOCK_M = 128, 128, 64
NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS = 3, 2
TOL = 5e-3

# (M, K_out, N_in): M is the GEMM contraction dim (reduced away in dW).
SHAPES = [
    (1024, 256, 256),
    (4096, 1024, 1024),
    (8192, 2048, 1024),
    (16384, 1024, 1024),
]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def make_inputs(shape):
    M, K_out, N_in = shape
    dout = torch.randn(M, K_out, device="cuda", dtype=torch.float16)
    act = torch.randn(M, N_in, device="cuda", dtype=torch.float16)
    # Separate dW/db outputs per side so gen and hw results stay comparable.
    dw = torch.full((K_out, N_in), float("nan"), device="cuda", dtype=torch.float16)
    db = torch.full((K_out,), float("nan"), device="cuda", dtype=torch.float32)
    dw_hw = torch.full((K_out, N_in), float("nan"), device="cuda", dtype=torch.float16)
    db_hw = torch.full((K_out,), float("nan"), device="cuda", dtype=torch.float32)
    return {
        "dout": dout, "act": act,
        "dw": dw, "db": db, "dw_hw": dw_hw, "db_hw": db_hw,
        "M": M, "K_out": K_out, "N_in": N_in,
    }


def gen_call(generated, inputs):
    generated.wgrad_bias_nows[(_num_sms(),)](
        inputs["dout"], inputs["act"], inputs["dw"], inputs["db"],
        inputs["M"], inputs["K_out"], inputs["N_in"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["dw"]


def hw_call(handwritten, inputs):
    handwritten.wgrad_bias_ws[(_num_sms(),)](
        inputs["dout"], inputs["act"], inputs["dw_hw"], inputs["db_hw"],
        inputs["M"], inputs["K_out"], inputs["N_in"],
        BLOCK_KO=BLOCK_KO, BLOCK_NI=BLOCK_NI, BLOCK_M=BLOCK_M,
        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
        num_warps=4, num_ctas=1,
    )
    return inputs["dw_hw"]


def metric(shape):
    M, K_out, N_in = shape
    return (2 * K_out * N_in * M, 1e12, "TFLOPS")  # bias reduce is negligible vs the GEMM


def reference(inputs):
    return (inputs["dout"].float().T @ inputs["act"].float()).to(torch.float16)
