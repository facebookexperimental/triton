"""TLX warp-spec tmem_load -> arrive -> reduce fusion compile-only test.

Covers T280816632: tcgen05.ld.red via ttng.tmem_load {redOp} must fuse
even when the load sits on a warp-spec partition boundary with an
immediately following arrive barrier. This is the compile-only version of
the runtime test in test_tlx_sm103_tmem_load_reduce.py: it compiles the
same kernel for sm103 and checks the fused instruction in PTX instead of
launching it. It runs on sm100 CI, where no sm103 hardware is available.
"""

# NOTE: delete this file once GB300 or Rubin is available in CI: the sm103
# runtime test will cover this pattern on real hardware, and this
# compile-only stand-in will no longer be needed.

import pytest

import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


# Duplicated from test_tlx_sm103_tmem_load_reduce.py; keep in sync.
# A Blackwell MMA feeding a row reduction. The reduction result must be
# stored -- an unused tl.max is DCE'd well before the fusion pass runs,
# which leaves no tmem_load+reduce pattern for it to match.
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


# Compile-only (no launch); gated to Blackwell just to reduce test bloat.
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_compile_only_tlx_tmem_load_reduce_with_arrive():
    """Compile-only: the runtime kernel fuses to tcgen05.ld.red for sm103."""
    src = ASTSource(
        fn=_tmem_rowmax_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "mx_ptr": "*fp32",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
    )
    k = triton.compile(src, target=GPUTarget("cuda", 103, 32), options={"num_warps": 4, "num_stages": 3})
    assert k.asm["cubin"] != b""
    assert "tcgen05.ld.red" in k.asm["ptx"], "expected tcgen05.ld.red in PTX"
