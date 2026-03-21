"""
Tests for multi-CTA reduction support in Triton.

Tests that the ``multi_cta=True`` parameter on ``tl.range`` correctly:
1. Emits the ``tt.multi_cta`` IR attribute on the ``scf.for`` loop
2. The MultiCTAReduction compiler pass detects and transforms the loop
3. Falls back to single-CTA behavior when cluster_dims == (1,1,1)
"""

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 1 : IR attribute emission
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


@triton.jit
def _kernel_with_multi_cta(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    _acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + row * N + cols, mask=cols < N, other=0.).to(tl.float32)
        _acc += x
    result = tl.sum(_acc, axis=0)
    tl.store(Y + row, result)


@triton.jit
def _kernel_without_multi_cta(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    _acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + row * N + cols, mask=cols < N, other=0.).to(tl.float32)
        _acc += x
    result = tl.sum(_acc, axis=0)
    tl.store(Y + row, result)


def test_multi_cta_ir_attribute():
    """Verify that multi_cta=True emits tt.multi_cta on the scf.for loop."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    #With multi_cta = True
    src = ASTSource(fn=_kernel_with_multi_cta, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=target)
    ttir = compiled.asm.get("ttir", "")
    assert "tt.multi_cta" in ttir, \
        f"Expected tt.multi_cta attribute in TTIR but not found:\n{ttir[:2000]}"

    #Without multi_cta — should NOT have the attribute
    src_no = ASTSource(fn=_kernel_without_multi_cta, signature=sig, constexprs=constexprs)
    compiled_no = triton.compile(src_no, target=target)
    ttir_no = compiled_no.asm.get("ttir", "")
    assert "tt.multi_cta" not in ttir_no, \
        "Unexpected tt.multi_cta attribute in TTIR without multi_cta=True"


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 2 : Single - CTA fallback(cluster_dims = 1, 1, 1)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


def test_multi_cta_single_cta_fallback():
    """When cluster_dims == (1,1,1), multi_cta=True should be a no-op."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    #Compile with default cluster_dims(1, 1, 1) — pass should strip the attr
    src = ASTSource(fn=_kernel_with_multi_cta, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=target)
    ttgir = compiled.asm.get("ttgir", "")
    #After the pass runs, tt.multi_cta should be removed
    assert "tt.multi_cta" not in ttgir, \
        f"tt.multi_cta should be removed after pass for single-CTA:\n{ttgir[:2000]}"


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 3 : Multi - CTA IR transformation(cluster_dims > 1)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


def test_multi_cta_generates_cluster_ops():
    """When cluster_dims > 1, the pass should generate cluster CTA ops."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_with_multi_cta, signature=sig, constexprs=constexprs)
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": 4, "cluster_dims": (1, 4, 1)},
    )
    ttgir = compiled.asm.get("ttgir", "")
    #After transformation, should see cluster CTA rank op
    assert "cluster_cta_id" in ttgir.lower() or "ClusterCTAId" in ttgir, \
        f"Expected cluster CTA id op in TTGIR for multi-CTA:\n{ttgir[:3000]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
