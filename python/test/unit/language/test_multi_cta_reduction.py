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
    #After transformation, should see cluster CTA rank op and loop partitioning
    assert "cluster_id" in ttgir.lower() or "nvgpu.cluster_id" in ttgir, \
        f"Expected cluster id op in TTGIR for multi-CTA:\n{ttgir[:3000]}"
    assert "arith.divui" in ttgir, \
        f"Expected arith.divui (loop partitioning) in TTGIR for multi-CTA:\n{ttgir[:3000]}"


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 4 : 2D block (BLOCK_SIZE_M rows) — IR attribute emission
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


@triton.jit
def _kernel_with_multi_cta_2d(
    X,
    Y,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    _acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE_N, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        ptrs = X + rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(ptrs, mask=mask, other=0.).to(tl.float32)
        _acc += x
    result = tl.sum(_acc, axis=1)
    tl.store(Y + rows, result, mask=rows < M)


def test_multi_cta_2d_block_ir_attribute():
    """Verify that multi_cta=True emits tt.multi_cta on 2D block kernel."""
    sig = {"X": "*fp32", "Y": "*fp32", "M": "i32", "N": "i32"}
    constexprs = {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_with_multi_cta_2d, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=target)
    ttir = compiled.asm.get("ttir", "")
    assert "tt.multi_cta" in ttir, \
        f"Expected tt.multi_cta attribute in TTIR for 2D block kernel but not found:\n{ttir[:2000]}"


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 5 : 2D block multi-CTA pass transformation(cluster_dims > 1)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


def test_multi_cta_2d_block_generates_cluster_ops():
    """When cluster_dims > 1, the pass should generate cluster CTA ops for 2D blocks."""
    sig = {"X": "*fp32", "Y": "*fp32", "M": "i32", "N": "i32"}
    constexprs = {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_with_multi_cta_2d, signature=sig, constexprs=constexprs)
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": 4, "cluster_dims": (1, 4, 1)},
    )
    ttgir = compiled.asm.get("ttgir", "")
    assert "cluster_id" in ttgir.lower() or "nvgpu.cluster_id" in ttgir, \
        f"Expected cluster id op in TTGIR for 2D multi-CTA block:\n{ttgir[:3000]}"
    assert "arith.divui" in ttgir, \
        f"Expected arith.divui (loop partitioning) in TTGIR for 2D multi-CTA block:\n{ttgir[:3000]}"


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 6 : Reject non-additive loop body (e.g., acc *= x)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


@triton.jit
def _kernel_mul_accumulation(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    _acc = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE, multi_cta=True):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + row * N + cols, mask=cols < N, other=1.).to(tl.float32)
        _acc *= x
    result = tl.sum(_acc, axis=0)
    tl.store(Y + row, result)


def test_multi_cta_rejects_mul_loop_body():
    """multi_cta=True with acc *= x should fail when cluster_dims > 1."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_mul_accumulation, signature=sig, constexprs=constexprs)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(
            src,
            target=target,
            options={"num_warps": 4, "cluster_dims": (1, 4, 1)},
        )


def test_multi_cta_mul_loop_body_ok_single_cta():
    """multi_cta=True with acc *= x should be fine when cluster_dims == (1,1,1)."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_mul_accumulation, signature=sig, constexprs=constexprs)
    # Single CTA: pass strips the attribute without validation, should succeed.
    compiled = triton.compile(src, target=target)
    ttgir = compiled.asm.get("ttgir", "")
    assert "tt.multi_cta" not in ttgir


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 7 : Reject non-additive reduce combiner (e.g., tl.max)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


@triton.jit
def _kernel_max_reduce(
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
    result = tl.max(_acc, axis=0)
    tl.store(Y + row, result)


def test_multi_cta_rejects_non_add_reduce_combiner():
    """multi_cta=True with tl.max reduce should fail when cluster_dims > 1."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_max_reduce, signature=sig, constexprs=constexprs)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(
            src,
            target=target,
            options={"num_warps": 4, "cluster_dims": (1, 4, 1)},
        )


def test_multi_cta_max_reduce_ok_single_cta():
    """multi_cta=True with tl.max reduce should be fine when cluster_dims == (1,1,1)."""
    sig = {"X": "*fp32", "Y": "*fp32", "N": "i32"}
    constexprs = {"BLOCK_SIZE": 1024}
    target = GPUTarget("cuda", 100, 32)

    src = ASTSource(fn=_kernel_max_reduce, signature=sig, constexprs=constexprs)
    compiled = triton.compile(src, target=target)
    ttgir = compiled.asm.get("ttgir", "")
    assert "tt.multi_cta" not in ttgir


#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Test 8 : Valid additive kernel still compiles with cluster_dims > 1
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


def test_multi_cta_additive_kernel_accepted():
    """multi_cta=True with acc += x and tl.sum should succeed with cluster_dims > 1."""
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
    assert "tt.multi_cta" not in ttgir, "tt.multi_cta should be consumed by the pass"
    assert "cluster_id" in ttgir.lower() or "nvgpu.cluster_id" in ttgir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
