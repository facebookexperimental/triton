"""
Test that the Insert2CTASync pass is correctly integrated into the compiler
pipeline and handles edge cases.

The core pass logic is tested by the MLIR lit test
(test/Hopper/TwoCTA/insert_2cta_sync.mlir). This Python test verifies:
1. The pass doesn't crash when cluster_dims >= 2 (integration test)
2. The pass is a no-op for single-CTA configs (negative test)

Full 2-CTA MMA testing requires tensor descriptors (TMA loads) which are
not supported via the ASTSource compilation path. Use the JIT path with
a Blackwell GPU for end-to-end 2-CTA correctness testing.
"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c)


SIG = {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "i32",
    "stride_bk": "i32",
    "stride_bn": "i32",
    "stride_cm": "i32",
    "stride_cn": "i32",
}
CONSTEXPRS = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}


def test_compile_with_cluster_dims_does_not_crash():
    """Verify that compiling with cluster_dims=(2,1,1) and num_ctas=2 does
    not crash. Without two_ctas=True on tl.dot(), PlanCTA uses its default
    N-split heuristic, and canUseTwoCTAs() does not trigger.

    For full 2-CTA e2e testing with two_ctas=True, use
    triton_addmm_2cta.py which uses TMA descriptor loads.
    """
    target = GPUTarget("cuda", 100, 32)
    src = ASTSource(fn=matmul_kernel, signature=SIG, constexprs=CONSTEXPRS)
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": 4, "num_stages": 2, "num_ctas": 2, "cluster_dims": (2, 1, 1)},
    )
    ttgir = compiled.asm.get("ttgir", "")
    # The pass should run without crashing. With pointer-based loads,
    # PlanCTA produces CTASplitNum=[1,2] not [2,1], so canUseTwoCTAs()
    # won't trigger — no cross-CTA sync ops expected.
    assert "tc_gen5_mma" in ttgir, "Expected tc_gen5_mma in TTGIR"
    assert "map_to_remote_buffer" not in ttgir, (
        "Pointer-based loads shouldn't trigger 2-CTA (need TMA descriptor loads)")


def test_no_2cta_sync_without_cluster():
    """With cluster_dims=(1,1,1), no cross-CTA sync should be inserted."""
    target = GPUTarget("cuda", 100, 32)
    src = ASTSource(fn=matmul_kernel, signature=SIG, constexprs=CONSTEXPRS)
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": 4, "num_stages": 2, "cluster_dims": (1, 1, 1)},
    )
    ttgir = compiled.asm.get("ttgir", "")
    assert "map_to_remote_buffer" not in ttgir, ("Unexpected map_to_remote_buffer in TTGIR without 2-CTA cluster")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
