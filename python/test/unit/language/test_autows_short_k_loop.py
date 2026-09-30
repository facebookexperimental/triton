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
def _pointer_a_persistent_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                               BN: tl.constexpr, BK: tl.constexpr, NUM_SMS: tl.constexpr, WS: tl.constexpr):
    b_desc = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(C, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    for tile_id in tl.range(tl.program_id(0), grid_m * grid_n, NUM_SMS, warp_specialize=WS):
        pid_m = tile_id % grid_m
        pid_n = tile_id // grid_m
        rm = pid_m * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            rk = ki * BK + tl.arange(0, BK)
            a = tl.load(A + rm[:, None].to(tl.int64) * K + rk[None, :]).to(tl.bfloat16)
            b = b_desc.load([ki * BK, pid_n * BN])
            acc = tl.dot(a, b, acc)
        c_desc.store([pid_m * BM, pid_n * BN], acc)


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


def _run_pointer_a_gemm(BN, K, num_stages):
    M, N, BM, BK, NUM_SMS = 1024, 1024, 128, 64, 148
    triton.set_allocator(_alloc)
    grid = (min(NUM_SMS, (M // BM) * (N // BN)), )
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        for it in range(5):
            g = torch.Generator(device="cuda").manual_seed(it)
            A = torch.randn(M, K, device="cuda", generator=g)
            B = torch.randn(K, N, device="cuda", generator=g).to(torch.bfloat16)
            outs = []
            for ws in (True, False):
                C = torch.full((M, N), float("nan"), device="cuda")
                k = _pointer_a_persistent_gemm[grid](A, B, C, M, N, K, BM, BN, BK, NUM_SMS, ws, num_warps=4,
                                                     num_stages=num_stages)
                outs.append(C)
                if ws:
                    kernel = k
            # Bitwise against the non-WS kernel, and close to the reference so
            # that both cannot be wrong together.
            assert torch.equal(outs[0], outs[1])
            ref = A.to(torch.bfloat16).float() @ B.float()
            torch.testing.assert_close(outs[0], ref, atol=1e-3, rtol=1e-5)
    return kernel


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("K, num_stages", [(128, 3)])
def test_autows_pointer_a_short_k_loop(K, num_stages):
    kernel = _run_pointer_a_gemm(128, K, num_stages)
    assert "ttg.warp_specialize(" in kernel.asm["ttgir"]


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
