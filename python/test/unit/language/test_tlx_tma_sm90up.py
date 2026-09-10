"""TLX tma tests -- Hopper and newer."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor


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

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
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

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
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
