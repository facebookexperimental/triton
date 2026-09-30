"""
End-to-end tests for K loops whose trip count is at most num_stages - 1 under
meta WS.

With TRITON_USE_META_WS the software pipeliner peels its epilogue. When the K
loop runs no more times than the prologue covers, the kernel loop runs zero
times and the first peeled MMA must read use_acc=false; otherwise it
accumulates onto whatever the accumulator buffer held before. Each test
launches several times so that the accumulator starts dirty.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell


@triton.jit
def _first_flag_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     STAGES: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.full((BM, BN), 7.0, tl.float32)
    first = True
    for ki in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
        a = a_desc.load([pid_m * BM, ki * BK])
        b = b_desc.load([ki * BK, pid_n * BN])
        acc = tl.dot(a, b, tl.where(first, 0.0, acc))
        first = False
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def _alloc(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("K, num_stages", [(64, 2), (64, 4), (128, 3), (192, 4), (256, 3)])
def test_short_k_loop_carried_constant_flag(K, num_stages):
    # The flag's yield is the constant false, which the peeled epilogue used
    # even when the kernel loop never ran.
    M, N, BM, BN, BK = 512, 512, 128, 128, 64
    triton.set_allocator(_alloc)
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        for it in range(3):
            g = torch.Generator(device="cuda").manual_seed(it)
            A = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
            B = torch.randn(K, N, device="cuda", generator=g).to(torch.bfloat16)
            C = torch.empty(M, N, device="cuda")
            _first_flag_gemm[(M // BM, N // BN)](A, B, C, M, N, K, BM, BN, BK, num_stages, num_warps=4)
            torch.testing.assert_close(C, A.float() @ B.float(), atol=1e-4, rtol=1e-5)
