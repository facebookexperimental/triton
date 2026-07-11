"""Benchmark spec for case7 (fused wgrad GEMM + bias reduce) consumed by
examples/testing/perf_engine.py.

Launch logic mirrors perf_generated_vs_handwritten.py. dW and dB are compared
as one concatenated fp32 vector (both are ~sqrt(M)-scale, so a shared relative
error denominator is fair).
"""

from __future__ import annotations

import torch

BLOCK_KO, BLOCK_NI, BLOCK_M = 128, 128, 64
NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS = 3, 2
TOL = 5e-3

SHAPES = [(4096, 1024, 1024), (8192, 2048, 1024), (16384, 1024, 1024)]


def make_inputs(shape):
    M, K_out, N_in = shape
    return {
        "dout": torch.randn(M, K_out, device="cuda", dtype=torch.float16),
        "act": torch.randn(M, N_in, device="cuda", dtype=torch.float16),
        "dw": torch.empty(K_out, N_in, device="cuda", dtype=torch.float16),
        "db": torch.empty(K_out, device="cuda", dtype=torch.float32),
        "M": M,
        "K_out": K_out,
        "N_in": N_in,
        "NUM_SMS": torch.cuda.get_device_properties(0).multi_processor_count,
    }


def _cat(dw, db):
    return torch.cat([dw.float().flatten(), db.float().flatten()])


def gen_call(generated, inputs):
    grid = (inputs["NUM_SMS"],)
    generated.wgrad_bias_nows[grid](
        inputs["dout"], inputs["act"], inputs["dw"], inputs["db"],
        inputs["M"], inputs["K_out"], inputs["N_in"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return _cat(inputs["dw"], inputs["db"])


def hw_call(handwritten, inputs):
    grid = (inputs["NUM_SMS"],)
    handwritten.wgrad_bias_ws[grid](
        inputs["dout"], inputs["act"], inputs["dw"], inputs["db"],
        inputs["M"], inputs["K_out"], inputs["N_in"],
        BLOCK_KO=BLOCK_KO, BLOCK_NI=BLOCK_NI, BLOCK_M=BLOCK_M,
        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
        num_warps=4, num_ctas=1,
    )
    return _cat(inputs["dw"], inputs["db"])


def metric(shape):
    M, K_out, N_in = shape
    return (2 * K_out * N_in * M, 1e12, "TFLOPS")


def reference(inputs):
    dout, act = inputs["dout"].float(), inputs["act"].float()
    return torch.cat(
        [torch.matmul(dout.T, act).flatten(), dout.sum(0).flatten()]
    )
