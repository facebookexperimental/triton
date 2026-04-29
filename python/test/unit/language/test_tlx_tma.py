import math
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.runtime.fbcode_gating import is_fbcode_dependant

if is_fbcode_dependant():
    from python.test.unit.language.conftest import _swizzle_scale_to_5d
else:
    from conftest import _swizzle_scale_to_5d


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("use_prefetch", [False, True])
def test_descriptor_load(use_prefetch, device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_load_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                               USE_PREFETCH: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        if USE_PREFETCH:
            tlx.async_descriptor_prefetch_tensor(desc_in, [off_m, off_n])
        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_load_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                          USE_PREFETCH=use_prefetch)
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_local_to_global") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_store_wait") == 1
    assert kernel.asm["ttgir"].count("ttng.fence_async_shared") == 1
    if use_prefetch:
        assert kernel.asm["ttgir"].count("ttng.async_tma_prefetch") == 1
        assert kernel.asm["ptx"].count("cp.async.bulk.prefetch.tensor") == 1
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_descriptor_load_prefetch_ws(device):
    """Test TMA prefetch in a warp-specialized kernel.

    Group 0 (consumer): arrives on smem_empty barrier, pretending it consumed the buffer.
    Group 1 (producer): prefetches the TMA tensor, waits for smem_empty, then issues the TMA load.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def prefetch_ws_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        smem_full = tlx.alloc_barriers(tl.constexpr(1))
        smem_full_bar = tlx.local_view(smem_full, 0)
        smem_empty = tlx.alloc_barriers(tl.constexpr(1))
        smem_empty_bar = tlx.local_view(smem_empty, 0)

        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        with tlx.async_tasks():
            with tlx.async_task("default"):
                # Consumer: pretend we consumed the buffer (e.g. through MMA), release smem_empty
                tlx.barrier_arrive(smem_empty_bar)

                # Wait for producer to fill the buffer
                tlx.barrier_wait(bar=smem_full_bar, phase=0)
                tlx.fence_async_shared()

                # Store the result back
                tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
                tlx.async_descriptor_store_wait(0)

            with tlx.async_task(num_warps=1):
                # Producer: prefetch, then wait for consumer to release buffer, then load
                # the descriptor and offsets should be identical to the actual async_descriptor_load
                tlx.async_descriptor_prefetch_tensor(desc_in, [off_m, off_n])

                tlx.barrier_wait(bar=smem_empty_bar, phase=0)

                tlx.barrier_expect_bytes(smem_full_bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)
                tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], smem_full_bar)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = prefetch_ws_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_prefetch") == 1
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 1
    assert kernel.asm["ptx"].count("cp.async.bulk.prefetch.tensor") == 1
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("level", ["L1", "L2"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_prefetch(level, use_mask, device):
    """Test pointer-based prefetch hint (tlx.prefetch)."""

    @triton.jit
    def prefetch_and_load_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        LEVEL: tl.constexpr,
        USE_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements if USE_MASK else None
        tlx.prefetch(input_ptr + offsets, level=LEVEL, mask=mask)
        x = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    BLOCK_SIZE = 1024
    n_elements = BLOCK_SIZE
    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (1, )
    kernel = prefetch_and_load_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, LEVEL=level, USE_MASK=use_mask)
    torch.testing.assert_close(x, y)
    assert "ttng.prefetch" in kernel.asm["ttgir"]
    assert f"prefetch.global.{level}" in kernel.asm["ptx"]


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("eviction_policy", ["evict_first", "evict_last", ""])
def test_descriptor_load_l2_cache_hint(eviction_policy, device):
    """Test that TMA loads can use L2 cache hints via eviction_policy parameter."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_load_kernel_with_cache_hint(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EVICTION_POLICY: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        # Use eviction_policy parameter for L2 cache hint
        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar, eviction_policy=EVICTION_POLICY)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_load_kernel_with_cache_hint[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                          BLOCK_SIZE_N=BLOCK_SIZE_N, EVICTION_POLICY=eviction_policy)

    # Verify the TMA load is present in IR
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1

    # Check that eviction policy is set in the IR (only for non-default policies)
    assert eviction_policy in kernel.asm["ttgir"]

    # Verify PTX output
    ptx = kernel.asm["ptx"]
    assert "cp.async.bulk.tensor" in ptx

    if eviction_policy:
        # Check for L2 cache policy creation and cache hint modifier
        assert "createpolicy.fractional.L2" in ptx
        assert "L2::cache_hint" in ptx
    else:
        # Normal/default policy should NOT have L2 cache hint
        assert "createpolicy.fractional.L2" not in ptx
        assert "L2::cache_hint" not in ptx

    # Verify correctness
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("eviction_policy", ["", "evict_first", "evict_last"])
def test_descriptor_store_l2_cache_hint(eviction_policy, device):
    """Test that TMA stores with L2 cache hint generate correct PTX."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_store_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EVICTION_POLICY: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        # Load without cache hint
        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        # Store with eviction policy
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n], eviction_policy=EVICTION_POLICY)
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_store_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                           EVICTION_POLICY=eviction_policy)

    # Verify the TMA store is present in IR
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_local_to_global") == 1
    if eviction_policy:
        assert f"evictionPolicy = {eviction_policy}" in ttgir

    # Verify PTX output
    ptx = kernel.asm["ptx"]
    assert "cp.async.bulk.tensor" in ptx
    if eviction_policy in ("evict_first", "evict_last"):
        # Should have L2 cache hint in PTX
        assert "createpolicy.fractional.L2" in ptx
        assert "L2::cache_hint" in ptx
    else:
        # Normal/default policy should NOT have L2 cache hint
        assert "createpolicy.fractional.L2" not in ptx
        assert "L2::cache_hint" not in ptx

    # Verify correctness
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("store_reduce", ["add", "min", "max"])
def test_descriptor_store_reduce(store_reduce, device):
    """Test that TMA stores with atomic reduction generate correct IR and produce correct results."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_store_reduce_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        STORE_REDUCE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 4)

        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n], store_reduce=STORE_REDUCE)
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.randint(1, 10, (M, N), dtype=torch.int32, device=device)
    if store_reduce == "add":
        y = torch.ones((M, N), dtype=torch.int32, device=device)
        expected = y + x
    elif store_reduce == "min":
        y = torch.full((M, N), 100, dtype=torch.int32, device=device)
        expected = torch.minimum(y, x)
    elif store_reduce == "max":
        y = torch.zeros((M, N), dtype=torch.int32, device=device)
        expected = torch.maximum(y, x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_store_reduce_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                  STORE_REDUCE=store_reduce)

    # Verify the TMA reduce is present in IR
    ttgir = kernel.asm["ttgir"]
    assert "async_tma_reduce" in ttgir

    # Verify PTX output contains the reduce instruction
    ptx = kernel.asm["ptx"]
    assert "cp.reduce.async.bulk.tensor" in ptx

    # Verify correctness
    torch.testing.assert_close(y, expected)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("eviction_policy", ["", "evict_first", "evict_last"])
def test_descriptor_store_reduce_l2_cache_hint(eviction_policy, device):
    """Test that TMA store-reduce with L2 cache hint generates correct PTX and produces correct results."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_store_reduce_l2_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EVICTION_POLICY: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 4)

        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n], store_reduce="add",
                                   eviction_policy=EVICTION_POLICY)
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.randint(1, 10, (M, N), dtype=torch.int32, device=device)
    y = torch.ones((M, N), dtype=torch.int32, device=device)
    expected = y + x
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_store_reduce_l2_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                     EVICTION_POLICY=eviction_policy)

    # Verify the TMA reduce is present in IR
    ttgir = kernel.asm["ttgir"]
    assert "async_tma_reduce" in ttgir
    if eviction_policy:
        assert f"evictionPolicy = {eviction_policy}" in ttgir

    # Verify PTX output
    ptx = kernel.asm["ptx"]
    assert "cp.reduce.async.bulk.tensor" in ptx
    if eviction_policy in ("evict_first", "evict_last"):
        # Should have L2 cache hint in PTX
        assert "createpolicy.fractional.L2" in ptx
        assert "L2::cache_hint" in ptx
    else:
        # Normal/default policy should NOT have L2 cache hint
        assert "createpolicy.fractional.L2" not in ptx
        assert "L2::cache_hint" not in ptx

    # Verify correctness
    torch.testing.assert_close(y, expected)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_descriptor_load_multicast(device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_load_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        CLUSTER_SIZE_M: tl.constexpr = 2
        cta_id = tlx.cluster_cta_rank()
        cta_id_m = cta_id % CLUSTER_SIZE_M
        cta_id_n = cta_id // CLUSTER_SIZE_M

        # have one CTA from each cluster row to initiate the TMA
        should_initiate_load = cta_id_m == cta_id_n

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N
        if should_initiate_load:
            # given CTA layout
            # [ 0, 2 ]
            # [ 1, 3 ]
            # for CTA 0: we want it to multicast to CTA 0 and 2
            # for CTA 3: we want it to multicast to CTA 1 and 3
            tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar,
                                      multicast_targets=[cta_id_m, cta_id_m + CLUSTER_SIZE_M])
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.rand((M, N), dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (2, 2)

    kernel = descriptor_load_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                          ctas_per_cga=(2, 2, 1))

    assert (kernel.asm["ptx"].count(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster") == 1)
    # x:
    # [ x0 | x2]
    # [ x1 | x3]
    # y:
    # [ y0 | y2]
    # [ y1 | y3]
    # we copied x0 to y0 and y2, x3 to y1 and y3. x1 and x2 are not copied.
    x0 = x[:64, :64]
    x3 = x[64:128, 64:128]

    y0 = y[:64, :64]
    y3 = y[64:128, 64:128]
    y1 = y[64:128, :64]
    y2 = y[:64, 64:128]

    torch.testing.assert_close(x0, y0)
    torch.testing.assert_close(x0, y2)
    torch.testing.assert_close(x3, y1)
    torch.testing.assert_close(x3, y3)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell for 2-CTA cluster with cta_group::2")
def test_descriptor_load_two_cta(device):
    """Test that async_descriptor_load with two_cta=True uses .cta_group::2.

    Two CTAs in a cluster each load their own tile independently. With two_cta=True,
    the TMA instruction uses .cta_group::2 so the mbarrier completion signal is
    automatically routed to the leader CTA's barrier based on %cluster_ctarank parity.
    The leader's barrier expects both CTAs' worth of bytes and only completes when
    both loads finish.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def two_cta_load_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        NUM_CTAS: tl.constexpr = 2
        cta_rank = tlx.cluster_cta_rank()
        is_leader = cta_rank == 0

        pid = tl.program_id(0)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // NUM_CTAS],
        )
        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // NUM_CTAS],
        )

        # Each CTA has its own SMEM buffer for its portion of the tile
        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N // NUM_CTAS), tl.float16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)

        # Leader's barrier tracks BOTH CTAs' TMA loads via cta_group::2
        bars = tlx.alloc_barriers(tl.constexpr(1), arrive_count=1)
        bar = tlx.local_view(bars, 0)

        TILE_BYTES: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N * tlx.size_of(tlx.dtype_of(desc_in))
        if is_leader:
            # Leader expects both CTAs' worth of bytes
            tlx.barrier_expect_bytes(bar, TILE_BYTES)
        tlx.cluster_barrier()

        # Cluster index: each cluster of NUM_CTAS CTAs processes one row tile
        cluster_id = pid // NUM_CTAS
        off_m = cluster_id * BLOCK_SIZE_M

        # Each CTA loads a portion of column-tile; cta_group::2 routes both
        # completions to the leader's barrier automatically
        off_n = cta_rank * BLOCK_SIZE_N // NUM_CTAS

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar, two_ctas=True)

        # Leader waits for both loads to complete
        if is_leader:
            tlx.barrier_wait(bar=bar, phase=0)

        # Cluster-wide sync: CTA 1 waits here until CTA 0 has confirmed both loads are done
        tlx.cluster_barrier()
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128, 128
    x = torch.rand((M, N), dtype=torch.float16, device=device)
    y = torch.zeros_like(x)
    grid = lambda meta: (2, )

    kernel = two_cta_load_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                       ctas_per_cga=(2, 1, 1))

    # Verify the PTX uses .cta_group::2
    ptx = kernel.asm["ptx"]
    assert ptx.count("cta_group::2") >= 1
    # Should NOT be multicast — each CTA loads its own tile
    assert "multicast::cluster" not in ptx

    # CTA 0 loaded x[0:128, 0:64] → y[0:128, 0:64]
    # CTA 1 loaded x[0:128, 64:128] → y[0:128, 64:128]
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_prefetch_tensormap(device):
    """Test that prefetch_tensormap emits prefetch.param.tensormap for a host-side descriptor."""

    @triton.jit
    def prefetch_tensormap_kernel_host_desc(in_desc, out_desc, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        tlx.prefetch(in_desc, tensormap=True)
        tlx.prefetch(out_desc, tensormap=True)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        tlx.async_descriptor_load(in_desc, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(out_desc, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    def test_host_desc():
        M, N = 128, 128
        BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
        x = torch.ones((M, N), dtype=torch.int16, device=device)
        y = torch.empty_like(x)

        in_desc = TensorDescriptor.from_tensor(x, [BLOCK_SIZE_M, BLOCK_SIZE_N])
        out_desc = TensorDescriptor.from_tensor(y, [BLOCK_SIZE_M, BLOCK_SIZE_N])
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        kernel = prefetch_tensormap_kernel_host_desc[grid](in_desc, out_desc, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                           BLOCK_SIZE_N=BLOCK_SIZE_N)
        # Make sure we're using generic address, not .param space
        assert kernel.asm["ptx"].count("prefetch.tensormap") == 2
        assert kernel.asm["ptx"].count("prefetch.param.tensormap") == 0
        torch.testing.assert_close(x, y)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def prefetch_tensormap_kernel_device_desc(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
        tlx.prefetch(desc_in, tensormap=True)
        tlx.prefetch(desc_out, tensormap=True)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence("async_shared")
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    def test_device_desc():
        triton.set_allocator(alloc_fn)
        M, N = 128, 128
        BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
        x = torch.ones((M, N), dtype=torch.int16, device=device)
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

        kernel = prefetch_tensormap_kernel_device_desc[grid](
            x,
            y,
            M,
            N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        # Make sure we're using generic address, not .param or even (unsupported) global space
        assert kernel.asm["ptx"].count("prefetch.tensormap") == 2
        assert kernel.asm["ptx"].count("prefetch.param.tensormap") == 0
        torch.testing.assert_close(x, y)

    test_host_desc()
    test_device_desc()


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_make_tensor_descriptor(device):
    """Test allocate_tensor_descriptor and make_tensor_descriptor together with TMA operations."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def kernel(input_ptr, output_ptr, SIZE, BLOCK_SIZE: tl.constexpr):
        # Allocate descriptor in global scratch memory using allocate_tensor_descriptor
        desc_ptrs = tlx.allocate_tensor_descriptor(num=2)

        # Create tensor descriptor using the global scratch pointer
        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptrs[0],
            base=input_ptr,
            shape=[SIZE],
            strides=[tl.constexpr(1)],
            block_shape=[BLOCK_SIZE],
        )

        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptrs[1],
            base=output_ptr,
            shape=[SIZE],
            strides=[tl.constexpr(1)],
            block_shape=[BLOCK_SIZE],
        )

        # Compute tile offset
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE

        # Load and store using standard descriptors
        # Reinterpret pointers as tensor descriptors
        desc_in = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptrs[0],
            block_shape=[BLOCK_SIZE],
            dtype=tlx.dtype_of(input_ptr),
        )
        desc_out = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptrs[1],
            block_shape=[BLOCK_SIZE],
            dtype=tlx.dtype_of(output_ptr),
        )
        x = desc_in.load([offset])
        desc_out.store([offset], x)

    triton.set_allocator(alloc_fn)
    SIZE = 128
    BLOCK_SIZE = 64
    x = torch.ones((SIZE, ), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(SIZE, BLOCK_SIZE), )

    compiled_kernel = kernel[grid](x, y, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    # Check that both global_scratch_alloc and tensormap_create were generated in IR
    ttgir = compiled_kernel.asm["ttgir"]
    assert ttgir.count("ttg.global_scratch_alloc") == 1, "Expected 1 global_scratch_alloc operation"
    assert ttgir.count("ttng.tensormap_create") == 2, "Expected 2 tensormap_create operations"
    assert ttgir.count("ttng.reinterpret_tensor_descriptor") == 2, "Expected 2 reinterpret_tensor_descriptor operations"

    # Verify the data was copied correctly through TMA operations
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_make_tensor_descriptor_mxfp8(device):
    """Test that encoding propagates from ReinterpretTensorDescOp back to MakeTensorDescOp with MXFP8 scales.

    When make_tensor_descriptor writes to a descPtr and reinterpret_tensor_descriptor
    reads from the same descPtr, the shared memory encoding from the TMA operation
    should propagate back to the make_tensor_descriptor operation.

    This test uses MXFP8 with 5D TMA scales to verify the encoding propagation in a realistic
    scaled GEMM scenario.
    """

    VEC_SIZE = 32  # mxfp8 uses 32 elements per scale factor

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def mxfp8_scaled_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        a_scale_ptr,
        b_scale_ptr,
        c_ptr,
        stride_cm,
        stride_cn,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate separate descriptor pointers for each descriptor
        desc_ptr_a = tlx.allocate_tensor_descriptor(num=1)
        desc_ptr_b = tlx.allocate_tensor_descriptor(num=1)
        desc_ptr_a_scale = tlx.allocate_tensor_descriptor(num=1)
        desc_ptr_b_scale = tlx.allocate_tensor_descriptor(num=1)

        # Create tensor descriptors and write to allocated pointers
        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptr_a[0],
            base=a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptr_b[0],
            base=b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_K, BLOCK_N],
        )

        # 5D scale descriptors: [1, rep_m/n, rep_k, 2, 256] for cuBLAS block scaling layout
        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptr_a_scale[0],
            base=a_scale_ptr,
            shape=[1, M // 128, K // 32 // 4, 2, 2 * 128],
            strides=[M // 128 * K // 32 // 4 * 2 * 2 * 128, K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptr_b_scale[0],
            base=b_scale_ptr,
            shape=[1, N // 128, K // 32 // 4, 2, 2 * 128],
            strides=[N // 128 * K // 32 // 4 * 2 * 2 * 128, K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        # Reinterpret the pointers as tensor descriptors
        desc_a = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptr_a[0],
            block_shape=[BLOCK_M, BLOCK_K],
            dtype=tl.float8e4nv,
        )
        desc_b = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptr_b[0],
            block_shape=[BLOCK_K, BLOCK_N],
            dtype=tl.float8e4nv,
        )
        # 5D reinterpret for scales
        desc_a_scale = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptr_a_scale[0],
            block_shape=[1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
            dtype=tl.uint8,
        )
        desc_b_scale = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptr_b_scale[0],
            block_shape=[1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
            dtype=tl.uint8,
        )

        # Allocate SMEM buffers
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float8e4nv, tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float8e4nv, tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8, tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8, tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)

        # Use reinterpreted descriptors for async loads
        tlx.async_descriptor_load(desc_a, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(desc_b, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(desc_a_scale, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(desc_b_scale, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tl.float16)

        # Store result
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    M, N, K = (128, 128, 256)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    # This matches cuBLAS block scaling layout used by tcgen5_mma_scaled
    a_scale = torch.randint(124, 130, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N, "M": M, "N": N, "K": K}
    kernel = mxfp8_scaled_kernel[(1, 1)](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        a_scale_5d,
        b_scale_5d,
        c,
        c.stride(0),
        c.stride(1),
        "e4m3",
        "e4m3",
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]

    # Verify that tensormap_create and reinterpret_tensor_descriptor operations are present
    assert ttgir.count("ttng.tensormap_create") == 4, (
        f"Expected 4 tensormap_create operations, found {ttgir.count('ttng.tensormap_create')}")
    assert ttgir.count("ttng.reinterpret_tensor_descriptor") == 4, (
        f"Expected 4 reinterpret_tensor_descriptor operations, found {ttgir.count('ttng.reinterpret_tensor_descriptor')}"
    )

    # Verify encoding propagation: tensormap_create should have shared memory encoding
    # The encoding propagates from ReinterpretTensorDescOp back to MakeTensorDescOp
    assert "#ttg.nvmma_shared" in ttgir or "#ttg.swizzled_shared" in ttgir, "Expected shared memory encoding in ttgir"

    # Compute reference
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("BLOCK_SIZE", [64])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_tensor_descriptor_ws_capture(BLOCK_SIZE, device):
    """Test that tensor descriptor parameters are properly captured in WS regions when used in inlined functions."""

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def load_helper(desc, offset):
        """Helper function that uses descriptor - will be inlined."""
        return desc.load([offset])

    @triton.jit
    def store_helper(desc, offset, data):
        """Helper function that stores using descriptor - will be inlined."""
        desc.store([offset], data)

    @triton.jit
    def kernel(input_ptr, output_ptr, SIZE, BLOCK_SIZE: tl.constexpr):
        # Create tensor descriptors
        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[SIZE],
            strides=[tl.constexpr(1)],
            block_shape=[BLOCK_SIZE],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[SIZE],
            strides=[tl.constexpr(1)],
            block_shape=[BLOCK_SIZE],
        )

        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE

        # Use tensor descriptor in WS regions with inlined function
        # The descriptor and its expanded parameters should be properly captured in non-default region
        with tlx.async_tasks(warp_specialize=True):
            with tlx.async_task("default"):
                # Default task does some trivial work
                dummy = pid + 1
                dummy = dummy * 2
            with tlx.async_task(num_warps=4):
                # Call helper functions that will be inlined in non-default region
                # The descriptor and its expanded parameters need to be captured from outer scope
                x = load_helper(desc_in, offset)
                store_helper(desc_out, offset, x)

    triton.set_allocator(alloc_fn)
    SIZE = 256
    input_data = torch.arange(SIZE, dtype=torch.float32, device=device)
    output_data = torch.zeros(SIZE, dtype=torch.float32, device=device)

    grid = lambda meta: (triton.cdiv(SIZE, BLOCK_SIZE), )
    kernel[grid](input_data, output_data, SIZE, BLOCK_SIZE)
    assert torch.allclose(output_data, input_data), "Tensor descriptor capture in WS region failed"
