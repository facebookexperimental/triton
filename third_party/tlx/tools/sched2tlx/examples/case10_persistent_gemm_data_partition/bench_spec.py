"""Benchmark spec for case2 (persistent GEMM) consumed by examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors run_generated.py (generated) and run_handwritten.py (reference).
Large shapes are included so the persistent outer loop runs many tiles — the exact
regime where the pre-fix emitter's name collision corrupts tile addressing.
"""

from __future__ import annotations

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

# Hand-written reference: BLOCK_* are constexpr launch params, kept at the
# classic 128-tile config.
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
# Generated kernel: the descriptor contract is baked into the emitted code.
# case2's pre_modulo.ttgir is a BM=BN=256 kernel that only lowers under the
# A.5 data partition (TRITON_DATA_PARTITION_N=auto picks n=2, m_size=128):
# a/b keep the ttgir's (256,64) load blocks; the partitioned epilogue stores
# per-group (m_size, BLOCK_N) = (128,256) chunks of C. See run_generated.py.
GEN_BLOCK_M, GEN_BLOCK_N, GEN_BLOCK_K = 256, 256, 64
GEN_STORE_M = 128  # m_size = GEN_BLOCK_M / applied_n
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
    # Separate output buffers per kernel: the harness compares gen vs hw
    # (rel_gen_vs_hw), so they must NOT alias — one shared `c` makes hw_call
    # overwrite gen's output and the comparison degenerate to _rel(c, c) == 0.
    c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    c_hw = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)
    b_t = b.t().contiguous()  # [N, K] so TMA loads [offs_bn, offs_k]
    # Reference (handwritten.py) descriptor set — 128-tile contract, writes c_hw.
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b_t, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(c_hw, [BLOCK_M, BLOCK_N])
    # Generated-kernel descriptor set — partitioned 256-tile contract, writes c.
    a_desc_gen = TensorDescriptor.from_tensor(a, [GEN_BLOCK_M, GEN_BLOCK_K])
    b_desc_gen = TensorDescriptor.from_tensor(b_t, [GEN_BLOCK_N, GEN_BLOCK_K])
    c_desc_gen = TensorDescriptor.from_tensor(c, [GEN_STORE_M, GEN_BLOCK_N])
    return {"a": a, "b": b, "c": c, "c_hw": c_hw,
            "a_desc": a_desc, "b_desc": b_desc, "c_desc": c_desc,
            "a_desc_gen": a_desc_gen, "b_desc_gen": b_desc_gen, "c_desc_gen": c_desc_gen,
            "M": M, "N": N, "K": K}


def gen_call(generated, inputs):
    grid = (_num_sms(),)
    generated.matmul_kernel_tma_persistent_simple[grid](
        inputs["a_desc_gen"], inputs["b_desc_gen"], inputs["c_desc_gen"],
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
    return inputs["c_hw"]


def metric(shape):
    M, N, K = shape
    return (2 * M * N * K, 1e12, "TFLOPS")


def reference(inputs):
    return torch.matmul(inputs["a"], inputs["b"])
