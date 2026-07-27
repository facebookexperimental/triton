"""Benchmark spec for case8 (multi-phase triple GEMM, general pooled path)
consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors bench_general.py's pooled side (committed generated.py):
three back-to-back GEMMs (fp16, fp16, bf16) in one kernel, one phase each,
with the phases' SMEM rings sharing one pooled backing. There is no
hand-written reference for this case, so the harness reports raw generated
throughput instead of a gen/hw ratio; the pool-vs-sum A/B and the kernel-SMEM
report live in bench_general.py.

Following the single-output contract of the other cases, the harness asserts
C1 (the first fp16 GEMM) against a fp32 torch reference; all three outputs
are validated by bench_general.py.
"""

from __future__ import annotations

import torch

TOL = 5e-3

# (M, N, K), shared by all three GEMMs; M and N must divide by the 128 tile.
SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def make_inputs(shape):
    M, N, K = shape
    return {
        "a1": torch.randn(M, K, device="cuda", dtype=torch.float16),
        "b1": torch.randn(K, N, device="cuda", dtype=torch.float16),
        "a2": torch.randn(M, K, device="cuda", dtype=torch.float16),
        "b2": torch.randn(K, N, device="cuda", dtype=torch.float16),
        "a3": torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
        "b3": torch.randn(K, N, device="cuda", dtype=torch.bfloat16),
        "c1": torch.empty(M, N, device="cuda", dtype=torch.float16),
        "c2": torch.empty(M, N, device="cuda", dtype=torch.float16),
        "c3": torch.empty(M, N, device="cuda", dtype=torch.bfloat16),
        "grid": (M // 128, N // 128),
        "M": M, "N": N, "K": K,
    }


def gen_call(generated, inputs):
    generated.triple_gemm_nows[inputs["grid"]](
        inputs["a1"], inputs["b1"], inputs["c1"],
        inputs["a2"], inputs["b2"], inputs["c2"],
        inputs["a3"], inputs["b3"], inputs["c3"],
        inputs["M"], inputs["N"], inputs["K"],
        num_warps=4, num_ctas=1, num_stages=2,
    )
    return inputs["c1"]


def hw_call(handwritten, inputs):
    raise NotImplementedError("case8 has no handwritten reference")


def metric(shape):
    M, N, K = shape
    return (3 * 2 * M * N * K, 1e12, "TFLOPS")  # three GEMMs


def reference(inputs):
    return torch.matmul(inputs["a1"].float(), inputs["b1"].float())
