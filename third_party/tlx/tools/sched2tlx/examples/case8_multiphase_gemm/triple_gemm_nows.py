"""Case 8 — multi-phase GEMM (THE multi-phase representative of the corpus).

One kernel, three sequential GEMM phases. Merges the coverage of the retired
case8_dual_gemm (homogeneous pair) and case9_hetero_dual_gemm (heterogeneous
phase) and is the first real N>2 exercise of the general multi-phase emitter:

  phase 0: C1 = A1 @ B1, fp16, BLOCK 128x128x128 — (128,128)xf16 rings
  phase 1: C2 = A2 @ B2, fp16, BLOCK 128x128x128 — homogeneous with phase 0
  phase 2: C3 = A3 @ B3, bf16, BLOCK 128x128x64  — heterogeneous rings/dtype

The three phases are time-disjoint per CTA, so their operand rings share one
pooled SMEM backing (smem_phase_group 0/1/2 under TRITON_MODULO_PHASE_POOL=
pool): footprint = max(phase pools), not the (way over budget) sum. The
feature matrix beyond this representative (spec-tree shape, rejection paths)
lives in the GPU-free unit tests test_multiphase_*.py under tools/sched2tlx.

Authoring constraints follow the modulo pre-dump recipe (case6 docstring):
descriptors + tl.dot, num_warps=4, loop nesting depth <= 1.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def triple_gemm_nows(
    A1, B1, C1, A2, B2, C2, A3, B3, C3,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K1: tl.constexpr, BLOCK_K3: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a1_desc = tl.make_tensor_descriptor(A1, [M, K], [K, 1], [BLOCK_M, BLOCK_K1])
    b1_desc = tl.make_tensor_descriptor(B1, [K, N], [N, 1], [BLOCK_K1, BLOCK_N])
    c1_desc = tl.make_tensor_descriptor(C1, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    a2_desc = tl.make_tensor_descriptor(A2, [M, K], [K, 1], [BLOCK_M, BLOCK_K1])
    b2_desc = tl.make_tensor_descriptor(B2, [K, N], [N, 1], [BLOCK_K1, BLOCK_N])
    c2_desc = tl.make_tensor_descriptor(C2, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    a3_desc = tl.make_tensor_descriptor(A3, [M, K], [K, 1], [BLOCK_M, BLOCK_K3])
    b3_desc = tl.make_tensor_descriptor(B3, [K, N], [N, 1], [BLOCK_K3, BLOCK_N])
    c3_desc = tl.make_tensor_descriptor(C3, [M, N], [N, 1], [BLOCK_M, BLOCK_N])

    # Phase 0: fp16, BK=128
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K1):
        a = a1_desc.load([off_m, k])
        b = b1_desc.load([k, off_n])
        acc1 = tl.dot(a, b, acc1)
    c1_desc.store([off_m, off_n], acc1.to(tl.float16))

    # Phase 1: fp16, BK=128 (homogeneous with phase 0)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K1):
        a = a2_desc.load([off_m, k])
        b = b2_desc.load([k, off_n])
        acc2 = tl.dot(a, b, acc2)
    c2_desc.store([off_m, off_n], acc2.to(tl.float16))

    # Phase 2: bf16, BK=64 (heterogeneous rings + dtype)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K3):
        a = a3_desc.load([off_m, k])
        b = b3_desc.load([k, off_n])
        acc3 = tl.dot(a, b, acc3)
    c3_desc.store([off_m, off_n], acc3.to(tl.bfloat16))
