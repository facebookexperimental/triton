"""Compile-only coverage for reduce-marker multi-CTA collectives."""

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

RUN_MULTI_CTA_REDUCTION_TESTS = False
pytestmark = pytest.mark.skipif(
    not RUN_MULTI_CTA_REDUCTION_TESTS,
    reason="Multi-CTA reduction tests are re-enabled in D123952662 after migration to reduce markers",
)


@triton.jit
def _marked_sum_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        cols = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + cols, mask=cols < n, other=0.0).to(tl.float32)
        acc += values

    result = tl.sum(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _unmarked_sum_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets, mask=offsets < n, other=0.0).to(tl.float32)
    result = tl.sum(values, axis=0)
    tl.store(y, result)


@triton.jit
def _marked_max_kernel(
    x,
    y,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    local_rank = tl.program_id(1) % NUM_CTAS
    acc = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + offsets, mask=offsets < n, other=-float("inf"))
        acc = tl.maximum(acc, values)

    result = tl.max(acc, axis=0, multi_cta=True)
    for off in tl.range(
            local_rank * BLOCK_SIZE,
            n,
            NUM_CTAS * BLOCK_SIZE,
    ):
        tl.store(y + off, result, mask=off < n)


@triton.jit
def _non_bool_marker_kernel(
    x,
    y,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets)
    result = tl.sum(values, axis=0, multi_cta=1)
    tl.store(y, result)


def _compile(fn, constexprs, cluster_dims=None):
    options = {"num_warps": 4}
    if cluster_dims is not None:
        options["cluster_dims"] = cluster_dims
    return triton.compile(
        ASTSource(
            fn=fn,
            signature={"x": "*fp32", "y": "*fp32", "n": "i32"} if "n" in fn.arg_names else {"x": "*fp32", "y": "*fp32"},
            constexprs=constexprs,
        ),
        target=GPUTarget("cuda", 100, 32),
        options=options,
    )


def test_reduce_marker_emitted_and_lowered():
    compiled = _compile(
        _marked_sum_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
    )
    assert "tt.multi_cta" in compiled.asm["ttir"]
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]
    assert "ttng.cluster_reduce" not in compiled.asm["ttgir"]


def test_unmarked_reduce_has_no_collective():
    compiled = _compile(_unmarked_sum_kernel, {"BLOCK_SIZE": 1024})
    assert "tt.multi_cta" not in compiled.asm["ttir"]
    assert "async_remote_shmem_store" not in compiled.asm["ttgir"]


def test_reduce_marker_single_cta_fallback():
    compiled = _compile(
        _marked_sum_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 1},
    )
    assert "tt.multi_cta" in compiled.asm["ttir"]
    assert "tt.multi_cta" not in compiled.asm["ttgir"]
    assert "async_remote_shmem_store" not in compiled.asm["ttgir"]


def test_marked_max_is_supported():
    compiled = _compile(
        _marked_max_kernel,
        {"BLOCK_SIZE": 1024, "NUM_CTAS": 4},
        cluster_dims=(1, 4, 1),
    )
    assert "async_remote_shmem_store" in compiled.asm["ttgir"]


def test_reduce_marker_requires_bool():
    with pytest.raises(Exception) as exc_info:
        _compile(_non_bool_marker_kernel, {"BLOCK_SIZE": 1024})
    cause = exc_info.value
    while cause is not None:
        if "multi_cta must be a bool" in str(cause):
            break
        cause = cause.__cause__
    assert cause is not None
