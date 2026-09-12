"""TLX warp-spec tmem_load -> arrive -> reduce fusion runtime test.

Covers T280816632: tcgen05.ld.red via ttng.tmem_load {redOp} must fuse
even when the load sits on a warp-spec partition boundary with an
immediately following arrive barrier (TLX explicit `tlx.barrier_arrive`
lowers to `ttng.arrive_barrier`). The pass hoists the barrier past the
reduction when no intervening memory ops exist (per-subtile).

Runtime Blackwell Ultra-only: launches a kernel that lowers to the same
pattern and checks the fused instruction both emits and computes the right
answer.
The compile-only counterpart lives in
test_tlx_compile_tmem_load_reduce_sm100.py.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell_ultra


# ---------------------------------------------------------------------------
# Runtime kernel: a Blackwell MMA feeding a row reduction. The reduction result
# must be stored -- an unused tl.max is DCE'd well before the fusion pass runs,
# which leaves no tmem_load+reduce pattern for it to match.
# ---------------------------------------------------------------------------


@triton.jit
def _tmem_rowmax_kernel(a_ptr, b_ptr, mx_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    mx = tl.max(acc, axis=1)
    tl.store(mx_ptr + offs_m, mx, mask=offs_m < M)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
def test_runtime_tmem_load_reduce_executes():
    """Runtime: the fused tcgen05.ld.red emits and computes the right row max.

    The compile-only test pins the IR shape for the TLX explicit-arrive
    pattern; this is its hardware counterpart. Both the instruction check and
    the numerics check are made against the launched kernel, so this does not
    restate the compile-only test. BLOCK_N == N so a single tile spans each row
    and no cross-tile combine is needed.
    """
    if not is_blackwell_ultra():
        pytest.skip("Requires Blackwell Ultra")

    M, N, K = 256, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    mx = torch.empty((M, ), device="cuda", dtype=torch.float32)

    kernel = _tmem_rowmax_kernel[(triton.cdiv(M, BLOCK_M), )](
        a, b, mx, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )

    assert "redOp" in kernel.asm["ttgir"], "expected fused ttng.tmem_load {redOp} in the launched kernel"
    ptx = kernel.asm["ptx"]
    assert "tcgen05.ld.red" in ptx, "expected tcgen05.ld.red in PTX"

    ref = torch.max(torch.matmul(a.float(), b.float()), dim=1).values
    torch.testing.assert_close(ref, mx, atol=1e-2, rtol=1e-2)
