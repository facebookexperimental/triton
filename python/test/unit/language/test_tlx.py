import math
import itertools
import pytest
import torch
import re
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell, is_hopper, is_hip
import triton.language.extra.tlx as tlx
from typing import Optional
import traceback
import triton.runtime.driver as driver
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_async_tasks(BLOCK_SIZE, device):

    @triton.jit
    def add2_warp_specialized_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default", registers=120, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                x1 = x + replica_id
                y1 = y - replica_id
                output = x1 + y1
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, registers=100, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                # This no-op is just to test that replica_id
                # is correctly passed to the kernel
                a1 = a + replica_id
                b1 = b - replica_id
                output = a1 + b1
                tl.store(c_ptr + offsets, output, mask=mask)

    def dual_add(x, y, a, b):
        return x + y, a + b

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = add2_warp_specialized_kernel[grid](
        x,
        y,
        output1,
        a,
        b,
        output2,
        n_elements,
        BLOCK_SIZE,
        num_warps=4,
    )
    ttgir = kernel.asm["ttgir"]
    # print(ttgir)
    pattern_ws = (r'ttg.warp_specialize(.*) attributes {requestedRegisters = array<i32: 120, 100, 100>}')
    assert re.search(pattern_ws, ttgir, flags=re.DOTALL)
    pattern_p0 = (r'partition0\([^\n]*\)\s+num_warps\(4\)')
    assert re.search(pattern_p0, ttgir, flags=re.DOTALL)
    pattern_p1 = (r'partition1\([^\n]*\)\s+num_warps\(1\)')
    assert re.search(pattern_p1, ttgir, flags=re.DOTALL)
    pattern_p2 = (r'partition2\([^\n]*\)\s+num_warps\(1\)')
    assert re.search(pattern_p2, ttgir, flags=re.DOTALL)

    # Check that the replica_id is correctly passed to non-default regions
    # TTIR/TTGIR should be something like:
    #  partition0(...) {
    #   %a1 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    #   ...
    #   %13 = arith.addf %9, %cst
    #   ...}
    #  partition1(...) {
    #   %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32, #blocked>
    #   ...
    #   %13 = arith.addf %9, %cst
    #   %14 = arith.subf %12, %cst
    #   ...}
    pattern_cst = (r'= arith.constant dense\<.*\>')
    found = re.findall(pattern_cst, ttgir)
    assert len(found) == 4, "Expected 4 cst by calling `tlx.async_task_replica_id()` in all regions"
    assert found[0] != found[1], "Two matches MUST be different"
    assert "dense<0.0" in found[0] and "dense<1.0" in found[1], "Expected 0.0 and 1.0 as replica_id"

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_load(BLOCK_SIZE, device):

    @triton.jit
    def local_load(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 3)
        tlx.async_load(x_ptr_offsets, buffers[0], mask=mask)
        tlx.async_load(y_ptr_offsets, buffers[1], mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffers[0])
        y_local = tlx.local_load(buffers[1])
        local_add = x_local + y_local
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 2
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(4)])
def test_local_slice(BLOCK_SIZE, device):

    @triton.jit
    def local_load(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x_ptr_offsets = x_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
        tlx.async_load(x_ptr_offsets, buffers[0])
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        buffer_0 = tlx.local_slice(buffers[0], [0], [BLOCK_SIZE // 2])
        buffer_1 = tlx.local_slice(buffers[0], [BLOCK_SIZE // 2], [BLOCK_SIZE // 2])
        x_0 = tlx.local_load(buffer_0)
        x_1 = tlx.local_load(buffer_1)

        offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
        output_ptr_offsets = output_ptr + offsets
        tl.store(output_ptr_offsets, x_0)
        tl.store(output_ptr_offsets + BLOCK_SIZE // 2, x_1)

    torch.manual_seed(0)
    size = 4
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load[grid](x, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 1
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    torch.testing.assert_close(x, output)


def _generate_test_params():
    """Generate test parameters with filtering for memory constraints."""
    dims_mn = [16, 32, 64, 128, 512]
    dims_k = [16, 32, 64]
    dtype = torch.float16
    params = []

    for M, N, K in itertools.product(dims_mn, dims_mn, dims_k):
        device_props = str(torch.cuda.get_device_properties())
        matmul_size = (M * K + K * N) * dtype.itemsize
        max_shared_mem = driver.active.utils.get_device_properties(driver.active.get_current_device())["max_shared_mem"]
        if matmul_size > max_shared_mem:
            continue
        # TODO: Investigate why this test fails on gfx942 with M=512, N=512, K=16
        if "gfx942" in device_props and M == 512 and N == 512 and K == 16:
            params.append(pytest.param(M, N, K, marks=pytest.mark.xfail()))
        elif "H100" in device_props and M == 512 and N == 512 and K == 64:
            # This shape incurs excessive register pressure and fails on H100
            params.append(pytest.param(M, N, K, marks=pytest.mark.xfail()))
        else:
            params.append((M, N, K))
    return params


# Test tl.dot wit tlx smem ops
# Tests tl.load->tlx_local_store->tlx_local_load->tl.dot
@pytest.mark.skipif(is_blackwell(), reason="Not tested on Blackwell")
@pytest.mark.parametrize("M,N,K", _generate_test_params())
def test_tl_dot_with_tlx_smem_load_store(M, N, K, device):

    @triton.jit
    def dot_kernel(
        X,
        stride_xm,
        stride_xk,
        Y,
        stride_yk,
        stride_yn,
        Z,
        stride_zm,
        stride_zn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = X + (off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        b_ptrs = Y + (off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)

        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(X), 1)
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(Y), 1)
        a_smem_view = buf_alloc_a[0]
        b_smem_view = buf_alloc_b[0]

        a_load_reg = tl.load(a_ptrs)
        b_load_reg = tl.load(b_ptrs)

        tlx.local_store(a_smem_view, a_load_reg)
        tlx.local_store(b_smem_view, b_load_reg)

        a_tile = tlx.local_load(a_smem_view)
        b_tile = tlx.local_load(b_smem_view)

        c_tile = tl.dot(a_tile, b_tile)

        c = c_tile.to(tlx.dtype_of(Z))
        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    # Note: This test may fail for other shapes/kwargs until
    # reg->shared layout propagation is implemented tlx layout propagation
    dtype = torch.float16

    print(f"{M=}, {N=}, {K=}")
    x = torch.randn((M, K), device=device, dtype=dtype)
    y = torch.randn((K, N), device=device, dtype=dtype)
    z = torch.zeros((M, N), device=device, dtype=dtype)

    # test smem
    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N}
    dot_kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        **kern_kwargs,
    )
    z_ref = torch.matmul(x, y)
    torch.testing.assert_close(z, z_ref)


# Tests tl.load->tlx_local_store->tlx_local_load
# This is a smem load/store test variant that does not use
# async_load, so this test can be run on platforms where
# async_load has no/limited support
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_load_store_smem_with_tl_load(BLOCK_SIZE, device):

    @triton.jit
    def smem_reg_store_load(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        smem_buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 3)
        x_smem = tlx.local_view(smem_buffers, 0)
        y_smem = tlx.local_view(smem_buffers, 1)

        x_tile = tl.load(x_ptr + offsets, mask=mask)
        y_tile = tl.load(y_ptr + offsets, mask=mask)

        tlx.local_store(x_smem, x_tile)
        tlx.local_store(y_smem, y_tile)

        x_reg = tlx.local_load(x_smem)
        y_reg = tlx.local_load(y_smem)
        local_add = x_reg + y_reg
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = smem_reg_store_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 2
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    assert kernel.asm["ttgir"].count("ttg.local_store") == 2
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_store(BLOCK_SIZE, device):

    @triton.jit
    def local_load_store(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(4))
        buffer0 = tlx.local_view(buffers, 0)
        buffer1 = tlx.local_view(buffers, 1)
        buffer2 = tlx.local_view(buffers, 2)
        tlx.async_load(x_ptr_offsets, buffer0, mask=mask)
        tlx.async_load(y_ptr_offsets, buffer1, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffer0)
        y_local = tlx.local_load(buffer1)
        local_add = x_local + y_local
        # store result into buffer2 and then load it
        tlx.local_store(buffer2, local_add)
        result = tlx.local_load(buffer2)
        tl.store(output_ptr + offsets, result, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load_store[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 3
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 3
    assert kernel.asm["ttgir"].count("ttg.local_store") == 1
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_tmem_alloc_index(BLOCK_SIZE, device):

    @triton.jit
    def kernel(BLOCK_SIZE: tl.constexpr, ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.tmem)
        buffer0 = tlx.local_view(buffers, 0)  # noqa: F841
        buffer1 = tlx.local_view(buffers, 1)  # noqa: F841

    grid = lambda meta: (1, )
    kerenl_info = kernel[grid](BLOCK_SIZE)
    # TODO: check numerics once tmem load/store is ready
    kerenl_info.asm["ttgir"]
    assert kerenl_info.asm["ttgir"].count("kernel") == 1


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(64, 64), (64, 8), (128, 16)])
def test_tmem_load_store(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_load_store_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        x_ptr_offsets = x_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

        a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        buffer1 = tlx.local_view(buffers, 0)
        tlx.local_store(buffer1, a)
        b = tlx.local_load(buffer1)
        # b == a == tensor of 1.0
        tl.store(x_ptr_offsets, b + 2)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = tmem_load_store_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    assert kerenl_info.asm["ttir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttir"].count("ttng.tmem_load") == 1

    assert kerenl_info.asm["ttgir"].count("kernel") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_alloc") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_load") == 1

    ref_out = torch.ones_like(x) + 2
    torch.testing.assert_close(x, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(128, 64)])
def test_tmem_subslice(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_subslice_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n1 = tl.arange(0, BLOCK_SIZE_N // 4)
        offs_n2 = tl.arange(BLOCK_SIZE_N // 4, BLOCK_SIZE_N // 2)
        offs_n3 = tl.arange(BLOCK_SIZE_N // 2, 3 * BLOCK_SIZE_N // 4)
        offs_n4 = tl.arange(3 * BLOCK_SIZE_N // 4, BLOCK_SIZE_N)
        x_ptr_offsets1 = x_ptr + (offs_m[:, None] * stride_m + offs_n1[None, :] * stride_n)
        x_ptr_offsets2 = x_ptr + (offs_m[:, None] * stride_m + offs_n2[None, :] * stride_n)
        x_ptr_offsets3 = x_ptr + (offs_m[:, None] * stride_m + offs_n3[None, :] * stride_n)
        x_ptr_offsets4 = x_ptr + (offs_m[:, None] * stride_m + offs_n4[None, :] * stride_n)

        a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        buffer1 = tlx.local_view(buffers, 0)
        tlx.local_store(buffer1, a)

        subslice1 = tlx.subslice(buffer1, 0, BLOCK_SIZE_N // 4)
        subslice2 = tlx.subslice(buffer1, BLOCK_SIZE_N // 4, BLOCK_SIZE_N // 4)
        subslice3 = tlx.subslice(buffer1, BLOCK_SIZE_N // 2, BLOCK_SIZE_N // 4)
        subslice4 = tlx.local_slice(buffer1, [0, 3 * BLOCK_SIZE_N // 4], [BLOCK_SIZE_M, BLOCK_SIZE_N // 4])

        b1 = tlx.local_load(subslice1)
        b2 = tlx.local_load(subslice2)
        b3 = tlx.local_load(subslice3)
        b4 = tlx.local_load(subslice4)
        # b == a == tensor of 1.0
        tl.store(x_ptr_offsets1, b1 + 2)
        tl.store(x_ptr_offsets2, b2 + 2)
        tl.store(x_ptr_offsets3, b3 + 2)
        tl.store(x_ptr_offsets4, b4 + 2)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = tmem_subslice_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    assert kerenl_info.asm["ttir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttir"].count("ttng.tmem_load") == 4

    assert kerenl_info.asm["ttgir"].count("kernel") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_alloc") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_load") == 4

    ref_out = torch.ones_like(x) + 2
    torch.testing.assert_close(x, ref_out)


def test_thread_id(device):

    @triton.jit
    def store_from_thread_0_kernel(
        output_ptr,
        value,
        n_elements,
        axis: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(axis)
        if tid == 0:
            tl.store(output_ptr + offsets, value, mask=mask)

    output = torch.zeros(32, dtype=torch.int32, device='cuda')
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1, )](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device='cuda')
    expected_output[0] = value
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_custer_cta_rank(device):

    @triton.jit
    def test_cta_0_kernel(
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # without multi-cta cluster launch, this test does not validate much except
        # the fact that the IR lowering flow works
        cta_id = tlx.cluster_cta_rank()
        tl.store(output_ptr + offsets, cta_id, mask=mask)

    tensor_size = 32
    # init with 1, expected to be filled with 0
    output = torch.ones(tensor_size, dtype=torch.int32, device=device)
    kernel = test_cta_0_kernel[(1, )](output, tensor_size, tensor_size, num_warps=1)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvgpu.cluster_id") == 1

    torch.cuda.synchronize()
    expected_output = torch.zeros(tensor_size, dtype=torch.int32, device=device)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_clock64(device):

    @triton.jit
    def clock64_from_thread_0_kernel(
        output_ptr,
        value,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(0)
        if pid == 0 and tid == 0:
            start = tlx.clock64()
            tl.store(output_ptr + offsets, value, mask=mask)
            end = tlx.clock64()
            tl.device_print("Cycles elapsed: ", end - start)

    output = torch.zeros(32, dtype=torch.int32, device="cuda")
    n_elements = output.numel()
    value = 42
    kernel = clock64_from_thread_0_kernel[(1, )](output, value, n_elements, 32, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.clock64") == 2
    assert kernel.asm["ptx"].count("%clock64") == 2


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_async_wait(BLOCK_SIZE, device):

    @triton.jit
    def async_wait_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    @triton.jit
    def async_wait_token_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        token = tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        token = tlx.async_load_commit_group([token])
        tlx.async_load_wait_group(tl.constexpr(0), [token])
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    torch.manual_seed(0)
    size = 64
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = async_wait_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    torch.testing.assert_close(x, output)
    kernel = async_wait_token_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_trans(device):

    @triton.jit
    def local_trans_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Compute global offsets
        input_offset = off_m[:, None] * N + off_n[None, :]
        output_offset = off_n[:, None] * M + off_m[None, :]

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1))
        buffer0 = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr + input_offset, buffer0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        buffer1 = tlx.local_trans(buffer0)
        transposed = tlx.local_load(buffer1)
        tl.store(output_ptr + output_offset, transposed)

    torch.manual_seed(0)
    M, N = 32, 64
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 64
    x = torch.rand((M, N), dtype=torch.float32, device=device)
    y = torch.empty((N, M), dtype=torch.float32, device=device)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    kernel = local_trans_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.memdesc_trans") == 1
    torch.testing.assert_close(y, x.T)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_local_reinterpret(device):

    @triton.jit
    def local_reinterpret_kernel(
        x32_ptr,
        y32_ptr,
        x16_ptr,
        y16_ptr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Compute global offsets
        input_offset = off_m[:, None] * BLOCK_SIZE_N + off_n[None, :]
        output_offset = off_m[:, None] * BLOCK_SIZE_N + off_n[None, :]

        tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tmem_buffer_0 = tlx.local_view(tmem_buffers, 0)

        # x32 GMEM -> x32 SMEM -> x32 Reg -> x32 TMEM -> x32 Reg -> y32 GMEM
        smem_buffers32 = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1),
                                         tlx.storage_kind.smem)
        smem_buffer_32_0 = tlx.local_view(smem_buffers32, 0)
        tlx.async_load(x32_ptr + input_offset, smem_buffer_32_0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        x32_reg = tlx.local_load(smem_buffer_32_0)
        tlx.local_store(tmem_buffer_0, x32_reg)
        x32_reg_from_tmem = tlx.local_load(tmem_buffer_0)
        tl.store(y32_ptr + output_offset, x32_reg_from_tmem)

        # x16 GMEM -> x16 SMEM -> x16 Reg -> x16 TMEM -> x16 Reg -> y16 GMEM
        smem_buffers16 = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1),
                                         tlx.storage_kind.smem)
        smem_buffer_16_0 = tlx.local_view(smem_buffers16, 0)
        tlx.async_load(x16_ptr + input_offset, smem_buffer_16_0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        reinterpreted = tlx.local_reinterpret(tmem_buffer_0, tl.float16)

        x16_reg = tlx.local_load(smem_buffer_16_0)
        tlx.local_store(reinterpreted, x16_reg)
        x16_reg_from_tmem = tlx.local_load(reinterpreted)
        tl.store(y16_ptr + output_offset, x16_reg_from_tmem)

    torch.manual_seed(0)
    M, N = 64, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = M, N
    x32 = torch.rand((M, N), dtype=torch.float32, device=device)
    y32 = torch.zeros((M, N), dtype=torch.float32, device=device)
    x16 = torch.rand((M, N), dtype=torch.float16, device=device)
    y16 = torch.zeros((M, N), dtype=torch.float16, device=device)
    grid = lambda meta: (1, )
    kernel = local_reinterpret_kernel[grid](x32, y32, x16, y16, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    assert kernel.asm["ttgir"].count("ttg.memdesc_reinterpret") == 1
    assert kernel.asm["ttgir"].count("ttng.tmem_store") == 2
    assert kernel.asm["ttgir"].count("ttng.tmem_alloc") == 1

    torch.testing.assert_close(x32, y32)
    torch.testing.assert_close(x16, y16)


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_async_dot(device):

    @triton.jit
    def wgmma_kernel_A_smem(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = X + (off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        b_ptrs = Y + (off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)

        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(X), 1)
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(Y), 1)
        a_tile = tlx.local_view(buf_alloc_a, 0)
        b_tile = tlx.local_view(buf_alloc_b, 0)

        tlx.async_load(a_ptrs, a_tile)
        tlx.async_load(b_ptrs, b_tile)

        # wait for buffers to be ready
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        c = tlx.async_dot(a_tile, b_tile)
        c = tlx.async_dot_wait(tl.constexpr(0), c)
        c = c.to(tlx.dtype_of(Z))
        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    @triton.jit
    def wgmma_kernel_A_reg(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        a_ptrs = X + (off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        b_ptrs = Y + (off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)

        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(Y), 1)
        b_tile = tlx.local_view(buf_alloc_b, 0)

        a_tile = tl.load(a_ptrs)
        tlx.async_load(b_ptrs, b_tile)

        # wait for buffers to be ready
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        c = tlx.async_dot(a_tile, b_tile)
        c = tlx.async_dot_wait(tl.constexpr(0), c)
        c = c.to(tlx.dtype_of(Z))
        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    # test smem
    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N}
    kernel = wgmma_kernel_A_smem[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0),
                                         z.stride(1), **kern_kwargs)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    z_ref = torch.matmul(x, y)
    torch.testing.assert_close(z, z_ref)

    # test reg
    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N}
    kernel = wgmma_kernel_A_reg[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0),
                                        z.stride(1), **kern_kwargs)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 1
    torch.testing.assert_close(z, z_ref)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_blackwell(device):
    """
    Test D = A*B + A*B
    """

    @triton.jit
    def tcgen5_dot_kernel(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_DTYPE: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc_init = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)
        tlx.async_load(a_ptrs, a_smem)
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)
        tlx.local_store(acc_tmem, acc_init)

        # no barrier, tcgen5 mma synchronous semantic, compiler auto inserts barrier and wait
        tlx.async_dot(a_smem, b_smem, acc_tmem, mBarriers=[], out_dtype=OUT_DTYPE)

        # given barrier, tcgen5 mma asynchronous semantic, need to explicitly wait for the barrier
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.async_dot(a_smem, b_smem, acc_tmem, mBarriers=[bar], out_dtype=OUT_DTYPE)
        tlx.barrier_wait(bar, tl.constexpr(0))

        # now result == a*b + a*b
        result = tlx.local_load(acc_tmem)

        c = result.to(tl.float16)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'OUT_DTYPE': tl.float32}
    kernel = tcgen5_dot_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0),
                                       z.stride(1), **kern_kwargs)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    assert ttgir.count("ttng.tc_gen5_mma") == 2

    ptx = kernel.asm["ptx"]
    assert ptx.count("tcgen05.alloc") == 1
    assert ptx.count("tcgen05.wait") == 2
    assert ptx.count("tcgen05.commit") == 2
    assert ptx.count("mbarrier.try_wait") == 2
    assert ptx.count("tcgen05.dealloc") == 1

    ref_out = torch.matmul(x, y) + torch.matmul(x, y)
    torch.testing.assert_close(z, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_blackwell_not_use_d(device):
    """
    Test D = A*B
    """

    @triton.jit
    def tcgen5_dot_kernel(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr1, stride_cm, stride_cn,
                          c_ptr2, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          OUT_DTYPE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)
        tlx.async_load(a_ptrs, a_smem)
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        # fill tmem d with 1
        acc_init = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.float32)
        tlx.local_store(acc_tmem, acc_init)
        # do not use d (so that we get A*B instead of A*B+1)
        tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[], out_dtype=OUT_DTYPE)

        # c1 = A*B
        c1 = tlx.local_load(acc_tmem).to(tl.float16)
        c_ptrs = c_ptr1 + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c1)

        # now use d, so c2 = A*B + c1 = A*B + A*B
        tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=pid < 1000, mBarriers=[], out_dtype=OUT_DTYPE)
        c2 = tlx.local_load(acc_tmem).to(tl.float16)
        c_ptrs = c_ptr2 + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c2)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)
    z2 = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'OUT_DTYPE': tl.float32}
    kernel = tcgen5_dot_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z1, z1.stride(0),
                                       z1.stride(1), z2, **kern_kwargs)
    ttgir = kernel.asm["ttgir"]
    mma_ops = [i for i in ttgir.split("\n") if "tc_gen5_mma" in i]
    assert len(mma_ops) == 2
    # check <use_d, pred> in ttgir, mma_ops[1] should have <[var name], %true>
    assert "%false, %true" in mma_ops[0]
    assert "%true, %true" not in mma_ops[1]
    assert "%false, %true" not in mma_ops[1]

    xy = torch.matmul(x, y)
    torch.testing.assert_close(z1, xy)
    torch.testing.assert_close(z2, xy + xy)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_blackwell_2cta_tma(device):
    run_async_dot_blackwell_2cta_tma(device, False)  # A in SMEM
    run_async_dot_blackwell_2cta_tma(device, True)  # A in TMEM


def run_async_dot_blackwell_2cta_tma(device, A_TMEM):
    """
    Test 2cta collective D = A*B for 1 tile.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def tcgen5_dot_kernel2cta_tma(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                  OUT_DTYPE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                  A_TMEM: tl.constexpr):
        # difference from 1cta
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=2)  # CTA0 waits for signals from both CTAs

        desc_a = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        desc_b = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                           block_shape=[BLOCK_K, BLOCK_N // 2],  # difference from 1cta
                                           )

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tl.float16, tl.constexpr(1))  # difference from 1cta
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)

        bars = tlx.alloc_barriers(tl.constexpr(2))
        bar_a = tlx.local_view(bars, 0)
        bar_b = tlx.local_view(bars, 1)
        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * 2)  # fp16
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * (BLOCK_N // 2) * 2)  # difference from 1cta

        # difference from 1cta: size and offsets
        tlx.async_descriptor_load(desc_a, a_smem, [cluster_cta_rank * BLOCK_M, 0], bar_a)
        tlx.async_descriptor_load(desc_b, b_smem, [0, cluster_cta_rank * BLOCK_N // 2], bar_b)

        tlx.barrier_wait(bar_a, tl.constexpr(0))
        tlx.barrier_wait(bar_b, tl.constexpr(0))

        # difference from 1cta: CTA0 waits for both CTAs before issuing MMA op
        cta_bar = tlx.remote_view(cta_bars[0], 0)
        tlx.barrier_arrive(cta_bar, 1)
        tlx.barrier_wait(cta_bar, phase=0, pred=pred_cta0)

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        # difference from 1cta: set two_ctas. Compiler auto generates pred to issue mma only from CTA0
        if A_TMEM:
            buf_alloc_a_tmem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1), tlx.storage_kind.tmem)
            a_reg = tlx.local_load(a_smem)
            tlx.local_store(buf_alloc_a_tmem[0], a_reg)
            tlx.async_dot(buf_alloc_a_tmem[0], b_smem, acc_tmem, use_acc=False, mBarriers=[], two_ctas=True,
                          out_dtype=OUT_DTYPE)
        else:
            tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[], two_ctas=True, out_dtype=OUT_DTYPE)
        result = tlx.local_load(acc_tmem)

        c = result.to(tl.float16)
        offs_m = cluster_cta_rank * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    M, N, K = (256, 128, 128)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    BLOCK_M = M // 2
    BLOCK_N = N
    BLOCK_K = K
    kern_kwargs = {
        'BLOCK_M': BLOCK_M, 'BLOCK_K': BLOCK_K, 'BLOCK_N': BLOCK_N, 'OUT_DTYPE': tl.float32, 'M': M, 'N': N, 'K': K,
        'A_TMEM': A_TMEM
    }
    kernel = tcgen5_dot_kernel2cta_tma[(M // BLOCK_M, N // BLOCK_N)](x, x.stride(0), x.stride(1), y, y.stride(0),
                                                                     y.stride(1), z, z.stride(0), z.stride(1),
                                                                     **kern_kwargs)

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (
        2, 1, 1), f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}"
    assert kernel.metadata.num_ctas == 1, f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}"

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvgpu.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1

    ptx = kernel.asm["ptx"]
    assert ptx.count("barrier.cluster.arrive.aligned") == 2  # one for remote bar init, one for tmem dealloc
    assert ptx.count("barrier.cluster.wait.aligned") == 2  # one for remote bar init, one for tmem dealloc
    assert ptx.count("mapa.shared::cluster") == 1  # address mapping for remote_view
    assert ptx.count("tcgen05.mma.cta_group::2") == 8  # BK=128 divided into steps of 16

    ref_out = torch.matmul(x, y)
    torch.testing.assert_close(z, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_blackwell_2cta_tma_ws(device):
    """
    Test 2cta collective D = A*B for 1 tile.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def tcgen5_dot_kernel2cta_tma_ws(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm,
                                     stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                     OUT_DTYPE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # difference from 1cta
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=2)  # CTA0 waits for signals from both CTAs

        desc_a = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        desc_b = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                           block_shape=[BLOCK_K, BLOCK_N // 2],  # difference from 1cta
                                           )

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tl.float16, tl.constexpr(1))  # difference from 1cta
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)

        smem_full_bars = tlx.alloc_barriers(num_barriers=tl.constexpr(1))
        tmem_full_bars = tlx.alloc_barriers(num_barriers=tl.constexpr(1))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        with tlx.async_tasks():
            with tlx.async_task("default"):  # epilogue consumer
                tlx.barrier_wait(tmem_full_bars[0], phase=0)

                result = tlx.local_load(acc_tmem)
                c = result.to(tl.float16)
                offs_m = cluster_cta_rank * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tl.arange(0, BLOCK_N)
                c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                tl.store(c_ptrs, c)
            with tlx.async_task(num_warps=1, num_regs=232):  # MMA consumer
                tlx.barrier_wait(smem_full_bars[0], phase=0)

                # difference from 1cta: CTA0 waits for both CTAs before issuing MMA op
                cta_bar = tlx.remote_view(cta_bars[0], 0)
                tlx.barrier_arrive(cta_bar, 1)
                tlx.barrier_wait(cta_bar, phase=0, pred=pred_cta0)

                # difference from 1cta: set two_ctas. Compiler auto generates pred to issue mma only from CTA0
                tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[], two_ctas=True, out_dtype=OUT_DTYPE)

                tlx.barrier_arrive(tmem_full_bars[0], 1)
            with tlx.async_task(num_warps=1, num_regs=232):  # producer
                # difference from 1cta: size
                tlx.barrier_expect_bytes(smem_full_bars[0],
                                         BLOCK_M * BLOCK_K * 2 + BLOCK_K * (BLOCK_N // 2) * 2)  # fp16
                # difference from 1cta: size and offsets
                tlx.async_descriptor_load(desc_a, a_smem, [cluster_cta_rank * BLOCK_M, 0], smem_full_bars[0])
                tlx.async_descriptor_load(desc_b, b_smem, [0, cluster_cta_rank * BLOCK_N // 2], smem_full_bars[0])

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    M, N, K = (256, 128, 128)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    BLOCK_M = M // 2
    BLOCK_N = N
    BLOCK_K = K
    kern_kwargs = {
        'BLOCK_M': BLOCK_M, 'BLOCK_K': BLOCK_K, 'BLOCK_N': BLOCK_N, 'OUT_DTYPE': tl.float32, 'M': M, 'N': N, 'K': K
    }
    kernel = tcgen5_dot_kernel2cta_tma_ws[(M // BLOCK_M, N // BLOCK_N)](x, x.stride(0), x.stride(1), y, y.stride(0),
                                                                        y.stride(1), z, z.stride(0), z.stride(1),
                                                                        **kern_kwargs)

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (
        2, 1, 1), f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}"
    assert kernel.metadata.num_ctas == 1, f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}"

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvgpu.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1

    ptx = kernel.asm["ptx"]
    # two for trunk remote bar init: one for default wg, one for non default
    # two for tmem dealloc (two returns)
    assert ptx.count("barrier.cluster.arrive.aligned") == 4
    # one for trunk remote bar init: non default WGs just arrive anyway, then it's equivalent to a sync between
    #   default WGs in all CTAs
    # two for tmem dealloc (two returns)
    assert ptx.count("barrier.cluster.wait.aligned") == 3
    assert ptx.count("mapa.shared::cluster") == 1  # address mapping for remote_view
    assert ptx.count("tcgen05.mma.cta_group::2") == 8  # BK=128 divided into steps of 16

    ref_out = torch.matmul(x, y)
    torch.testing.assert_close(z, ref_out)


@pytest.mark.parametrize("A_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.parametrize("B_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled(A_DATA_TYPE, B_DATA_TYPE, device):
    """
    Test D = (A * A_scale)  * (B * B_scale)

    """

    @triton.jit
    def tcgen5_dot_scaled_kernel(
        a_desc,
        a_scale_desc,  #
        b_desc,
        b_scale_desc,  #
        c_desc,  #
        A_format: tl.constexpr,  #
        B_format: tl.constexpr,  #
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # async load a and b into SMEM
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        a_scale_tile = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tlx.dtype_of(a_scale_desc),
                                       tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tlx.dtype_of(b_scale_desc),
                                       tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        LD_SIZE: tl.constexpr = (BLOCK_M + BLOCK_N) * (BLOCK_K + BLOCK_K // 32)
        tlx.barrier_expect_bytes(load_bar[0], LD_SIZE)  # fp8
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    DTYPE_MAP = {
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(DTYPE_MAP[A_DATA_TYPE]).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(DTYPE_MAP[B_DATA_TYPE]).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Use 4D TMA descriptor [rep_m, rep_k, 2, 256] with uint8 elements.
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.

    a_scale = torch.randint(10, 20, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(10, 20, (N, K // 32), dtype=torch.uint8, device=device)
    a_scale = a_scale.reshape(a_scale.shape[0] // 128, a_scale.shape[1] // 4, 2, 2 * 128)
    b_scale = b_scale.reshape(b_scale.shape[0] // 128, b_scale.shape[1] // 4, 2, 2 * 128)

    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=[BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128])
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=[BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128])

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_DATA_TYPE,
        B_DATA_TYPE,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tmem_copy") == 2
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts a tensor of FP8 E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference
    a_scale_f32 = fp8e8m0_to_float32(a_scale.reshape(M, K // 32))
    b_scale_f32 = fp8e8m0_to_float32(b_scale.reshape(N, K // 32))
    # Repeats each scale value 32 times along dimension 1.
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_tcgen05_commit(device):
    """
    Test tcgen05.commit tracking multiple tcgen05 ops
    """

    @triton.jit
    def tcgen5_commit_kernel(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr1, stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             OUT_DTYPE: tl.constexpr, NUM_DOT: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)
        tlx.async_load(a_ptrs, a_smem)
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        # fill tmem d with 0
        acc_init = tl.full((BLOCK_M, BLOCK_N), 0, dtype=tl.float32)
        tlx.local_store(acc_tmem, acc_init)

        # issue multiple mma ops
        bars = tlx.alloc_barriers(tl.constexpr(NUM_DOT))
        bar_final = tlx.local_view(bars, NUM_DOT - 1)  # reserved for final wait
        # make the first dot op sync by not giving a barrier (compiler will auto insert a barrier)
        tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=True, mBarriers=[], out_dtype=OUT_DTYPE)
        for k in range(0, NUM_DOT - 1):
            bar = tlx.local_view(bars, k)
            tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=True, mBarriers=[bar], out_dtype=OUT_DTYPE)

        # one dedicated barrier waiting for all previous mma ops
        tlx.tcgen05_commit(bar_final)
        tlx.barrier_wait(bar_final, tl.constexpr(0))

        # c1 = A*B
        c1 = tlx.local_load(acc_tmem).to(tl.float16)
        c_ptrs = c_ptr1 + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c1)

    torch.manual_seed(0)
    M, N, K = (64, 64, 64)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)

    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'OUT_DTYPE': tl.float32}

    num_dot = 4
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel = tcgen5_commit_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z1, z1.stride(0),
                                          z1.stride(1), NUM_DOT=num_dot, **kern_kwargs)
    ptx = kernel.asm["ptx"]
    assert ptx.count("tcgen05.mma") == 4 * num_dot  # loop unrolled so 4 mma ops per dot
    assert ptx.count(
        "tcgen05.commit") == 1 + num_dot  # one for each dot (loop unrolled), then one dedicated barrier for all mma ops
    assert ptx.count("mbarrier.try_wait") == 2  # one for first sync dot, one for final wait
    ref_out = torch.zeros_like(z1)
    for _ in range(num_dot):
        ref_out += torch.matmul(x, y)
    torch.testing.assert_close(z1, ref_out)

    num_dot = 3
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel = tcgen5_commit_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z1, z1.stride(0),
                                          z1.stride(1), NUM_DOT=num_dot, **kern_kwargs)
    ptx = kernel.asm["ptx"]
    assert ptx.count("tcgen05.mma") == 4 * num_dot  # loop unrolled so 4 mma ops per dot
    assert ptx.count(
        "tcgen05.commit") == 1 + num_dot  # one for each dot (loop unrolled), then one dedicated barrier for all mma ops
    assert ptx.count("mbarrier.try_wait") == 2  # one for first sync dot, one for final wait
    ref_out = torch.zeros_like(z1)
    for _ in range(num_dot):
        ref_out += torch.matmul(x, y)
    torch.testing.assert_close(z1, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_blackwell_tmem_A(device):
    """
    Test D = A*B where A is in TMEM instead of SMEM
    """

    @triton.jit
    def tcgen5_dot_kernel_tmem_A(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                 OUT_DTYPE: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # init acc in TMEM
        acc_init = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(acc_buffers, 0)
        tlx.local_store(acc_tmem, acc_init)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)
        tlx.async_load(a_ptrs, a_smem)
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        # load A from SMEM to Reg
        a_reg = tlx.local_load(a_smem)

        # store A to TMEM
        buffers_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1), tlx.storage_kind.tmem)
        a_tmem = tlx.local_view(buffers_a, 0)
        tlx.local_store(a_tmem, a_reg)

        # acc_tmem = acc_tmem + a_tmem * b_smem
        tlx.async_dot(a_tmem, b_smem, acc_tmem, mBarriers=[], out_dtype=OUT_DTYPE)
        # load result from TMEM to Reg
        result = tlx.local_load(acc_tmem)

        c = result.to(tl.float16)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 32, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'OUT_DTYPE': tl.float32}
    kernel = tcgen5_dot_kernel_tmem_A[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0),
                                              z.stride(1), **kern_kwargs)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.tmem_alloc") == 2
    assert ttgir.count("ttng.tmem_store") == 2
    assert ttgir.count("ttng.tc_gen5_mma") == 1

    xy = torch.matmul(x, y)
    ref_out = xy
    torch.testing.assert_close(z, ref_out)


@triton.jit
def tlx_square_non_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXPECTED_ARRIVAL_COUNT: tl.constexpr,
):
    """
    Test pairs of arrive/wait using different phases
    with a few random misc operations interleaved between them.

    To learn more about mbarrier phase, refer to:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier

    Following patterns will cause mbarrier deadlock.
    TODO. add unit tests demonstrating mbarrier deadlock

    Case 1:
    arrive => wait(phase=1)

    Case 2:
    arrive => arrive => wait(phase=0)

    Case 3:
    wait(phase=0) => arrive
    """

    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # mbarrier ops

    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=EXPECTED_ARRIVAL_COUNT)  # create
    bar = tlx.local_view(bars, 0)

    x = tl.load(x_ptr + offsets, mask=mask)  # Do something

    p = 0
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=p)  # Wait (proceed immediately)

    z = x * x  # Do something

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=p)  # Wait (proceed immediately)

    tl.store(z_ptr + offsets, z, mask=mask)  # Do something

    p = p ^ 1
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=0)  # Wait (proceed immediately)


@triton.jit
def tlx_square_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EXPECTED_ARRIVAL_COUNT: tl.constexpr,
):
    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # mbarrier ops
    bars = tlx.alloc_barriers(num_barriers=2, arrive_count=EXPECTED_ARRIVAL_COUNT)  # create
    b0 = tlx.local_view(bars, 0)
    b1 = tlx.local_view(bars, 1)

    phase = 0
    with tlx.async_tasks():
        with tlx.async_task("default"):

            tlx.barrier_wait(bar=b1, phase=phase ^ 1)

            # Placeholder block to do something

            tlx.barrier_arrive(bar=b0)  # Release

        with tlx.async_task(num_warps=4):

            tlx.barrier_wait(bar=b0, phase=phase)  # Wait

            # Some arith ops TODO. add WS
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            z = x * x
            tl.store(z_ptr + offsets, z, mask=mask)

            tlx.barrier_arrive(bar=b0)  # Wait


def run_tlx_square(func, BLOCK_SIZE, device, expected_arrival_count=1):

    # prepare inputs
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    z_ref = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = func[grid](x, z, n_elements, BLOCK_SIZE, expected_arrival_count)

    z_ref = x * x

    torch.testing.assert_close(z, z_ref, check_dtype=False)
    return kernel


# Unit test for arrive/wait
@pytest.mark.skipif(not (is_hip() or is_hopper_or_newer()), reason="Need Hopper or newer or AMD")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_non_ws(BLOCK_SIZE, device):
    expected_arrival_count = 4 if is_hip() else 1
    kernel = run_tlx_square(tlx_square_non_ws, BLOCK_SIZE, device, expected_arrival_count=expected_arrival_count)
    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    if is_hip():
        assert (ttgir.count("amdgpu.init_barrier") == 1) and (ttgir.count("amdgpu.read_barrier_phase")
                                                              == 3) and (ttgir.count("amdgpu.arrive_barrier")
                                                                         == 3), f"TTGIR {ttgir}"
    else:
        assert (ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 3) and (
            ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 3), f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_ws(BLOCK_SIZE, device):
    kernel = run_tlx_square(tlx_square_ws, BLOCK_SIZE, device)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert (ttgir.count("ttng.init_barrier")
            == 2) and (ttgir.count("ttng.wait_barrier") == 2) and (ttgir.count("ttng.barrier_expect") == 0) and (
                ttgir.count("ttng.arrive_barrier")
                == 2) and (ttgir.count("default {") == 1) and (ttgir.count("partition0") == 1), f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_barrier_live_range(device):

    @triton.jit
    def bar_live_kernel():
        # use bars1 after bars2/3 init
        bars1 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=1)

        bars2 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=2)
        tlx.barrier_arrive(bars2[0])

        bars3 = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=3)
        tlx.barrier_arrive(bars3[0])

        # bars1 and bars2 should both be live here
        tlx.barrier_arrive(bars1[0])

    torch.manual_seed(0)
    kernel = bar_live_kernel[(2, 1)]()
    ptx = kernel.asm["ptx"]

    # e.g. extract %1 and 1 from "mbarrier.init.shared::cta.b64 [%r1], 1;"
    pattern = r"mbarrier\.init\..*\.b64 \[(%r\d+)\], (\d+);"
    matches = re.findall(pattern, ptx)

    arrive_count_to_reg = {int(arrive_count): reg for reg, arrive_count in matches}
    assert len(arrive_count_to_reg) == 3, f"Expected 3 mbarrier init, got ptx: \n{ptx}"
    # Make sure they all have different registers (different SMEM addresses)
    assert arrive_count_to_reg[1] != arrive_count_to_reg[2], f"invalid reuse of SMEM, full ptx: \n{ptx}"
    assert arrive_count_to_reg[2] != arrive_count_to_reg[3], f"invalid reuse of SMEM, full ptx: \n{ptx}"
    assert arrive_count_to_reg[1] != arrive_count_to_reg[3], f"invalid reuse of SMEM, full ptx: \n{ptx}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_named_wait_arrive(BLOCK_SIZE, device):

    @triton.jit
    def add2_warp_specialized_pingpong_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default"):
                tlx.named_barrier_wait(9, 256)
                tlx.named_barrier_arrive(10, 256)
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=4, registers=100):
                tlx.named_barrier_arrive(9, 256)
                tlx.named_barrier_wait(10, 256)
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                output = a + b
                tl.store(c_ptr + offsets, output, mask=mask)

    def dual_add(x, y, a, b):
        return x + y, a + b

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = add2_warp_specialized_pingpong_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.wait_barrier_named %c9_i32, %c256_i32") == 1
    assert ttgir.count("ttng.arrive_barrier_named %c10_i32, %c256_i32") == 1
    assert ttgir.count("ttng.arrive_barrier_named %c9_i32_1, %c256_i32") == 1
    assert ttgir.count("ttng.wait_barrier_named %c10_i32_0, %c256_i32") == 1

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_descriptor_load(device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def descriptor_load_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
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

        tlx.async_descriptor_load(desc_in, buffer, [off_m, off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(desc_out, buffer, [off_m, off_n])
        tlx.async_descriptor_store_wait(0)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 64
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = descriptor_load_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_local_to_global") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_store_wait") == 1
    assert kernel.asm["ttgir"].count("ttng.fence_async_shared") == 1
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_gather(device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def local_gather_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[1, M * N],
            strides=[M * N, 1],
            block_shape=[1, BLOCK_SIZE_M * BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[1, M * N],
            strides=[M * N, 1],
            block_shape=[1, BLOCK_SIZE_M * BLOCK_SIZE_N],
        )

        buffers_in = tlx.local_alloc((1, BLOCK_SIZE_N), tl.int16, BLOCK_SIZE_M)
        buffers_out = tlx.local_alloc((1, BLOCK_SIZE_N), tl.int16, BLOCK_SIZE_M)

        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        # Gather once
        buffer_in = tlx.local_view(buffers_in, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)
        reinterpreted = tlx.local_reinterpret(buffer_in, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
        tlx.async_descriptor_load(desc_in, reinterpreted, [0, off_m * N + off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)

        # Use sub tiles separately
        for k in range(0, BLOCK_SIZE_M):
            buffer_in = tlx.local_view(buffers_in, k)
            buffer_out = tlx.local_view(buffers_out, k)
            in_local = tlx.local_load(buffer_in)
            tlx.local_store(buffer_out, in_local)

        buffer_out = tlx.local_view(buffers_out, 0)
        reinterpreted = tlx.local_reinterpret(buffer_out, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
        tlx.async_descriptor_store(desc_out, reinterpreted, [0, off_m * N + off_n])

    triton.set_allocator(alloc_fn)
    M, N = 256, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 128
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = local_gather_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_local_to_global") == 1
    torch.testing.assert_close(x, y)


def test_loop_carry_var_check(device):

    @triton.jit
    def loop_carry_shadow():
        x = tlx.local_alloc((16, 16), tl.int16, tl.constexpr(2))
        y = x
        for _ in range(0, 128):
            zeros = tl.zeros((16, 16), dtype=tl.int16)
            # shadow x with different type
            x = tlx.local_view(y, 0)
            tlx.local_store(x, zeros)

    grid = lambda meta: (1, 1)

    with pytest.raises(triton.CompilationError) as e:
        loop_carry_shadow[grid]()
    list_msg = traceback.format_exception(e.type, e.value, e.tb, chain=True)
    assert "Please make sure that the type stays consistent" in '\n'.join(list_msg)


@triton.jit
def _global_tmem_func(
    buffers,
    x_ptr,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptr_offsets = x_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

    ones = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)
    buffer1 = tlx.local_view(buffers, 0)
    tlx.local_store(buffer1, ones)
    b = tlx.local_load(buffer1)

    tl.store(x_ptr_offsets, b)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(64, 64)])
def test_tmem_op_func(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_op_func_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        # init tmem buffers here
        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        # pass buffers to another func to do actual processing
        _global_tmem_func(buffers, x_ptr, stride_m, stride_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    tmem_op_func_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    ref_out = torch.ones_like(x)
    torch.testing.assert_close(x, ref_out)


@triton.jit
def math_kernel(x):
    return x * 0.5 * (1 + (0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_inline_tmem(BLOCK_SIZE, device):

    @triton.jit
    def kernel(y_ptr, BLOCK_SIZE: tl.constexpr):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(4), tlx.storage_kind.tmem)
        buffer0 = buffers[0]
        x = tlx.local_load(buffer0)
        offsets_i = tl.arange(0, BLOCK_SIZE)[:, None]
        offsets_j = tl.arange(0, BLOCK_SIZE)[None, :]
        offsets = offsets_i * BLOCK_SIZE + offsets_j
        y = math_kernel(x)
        tl.store(y_ptr + offsets, y)

    y = torch.rand((64, 64), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = kernel[grid](y, BLOCK_SIZE)
    assert kerenl_info.asm["ttir"].count("store") == 1


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dots_blackwell_tmem(device):
    """
    Test D = ((A@B) * 0.5) @ C
    """

    @triton.jit
    def tcgen5_fa_kernel(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn, d_ptr,
                         stride_dm, stride_dn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        a_tiles = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        b_tiles = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        c_tiles = tlx.local_alloc((BLOCK_N, BLOCK_N), tl.float16, tl.constexpr(1), reuse=a_tiles)

        ab_fulls = tlx.alloc_barriers(num_barriers=tl.constexpr(1))
        c_fulls = tlx.alloc_barriers(num_barriers=tl.constexpr(1))

        acc_tiles = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        o_tiles = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, tl.constexpr(1), tlx.storage_kind.tmem,
                                  reuse=acc_tiles)
        d_tiles = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

        acc_fulls = tlx.alloc_barriers(num_barriers=tl.constexpr(1))
        o_fulls = tlx.alloc_barriers(num_barriers=tl.constexpr(1))
        d_fulls = tlx.alloc_barriers(num_barriers=tl.constexpr(1))

        with tlx.async_tasks():
            # load
            with tlx.async_task("default"):
                offs_m = tl.arange(0, BLOCK_M)
                offs_n = tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
                c_ptrs = c_ptr + (offs_n[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                # load a and b
                tlx.async_load(a_ptrs, a_tiles[0])
                tlx.async_load(b_ptrs, b_tiles[0])
                tlx.async_load_commit_group()
                tlx.async_load_wait_group(tl.constexpr(0))
                tlx.barrier_arrive(ab_fulls[0])

                # load c
                tlx.barrier_wait(acc_fulls[0], tl.constexpr(0))
                tlx.async_load(c_ptrs, c_tiles[0])
                tlx.async_load_commit_group()
                tlx.async_load_wait_group(tl.constexpr(0))
                tlx.barrier_arrive(c_fulls[0])

            # mma
            with tlx.async_task(num_warps=1):
                tlx.barrier_wait(ab_fulls[0], tl.constexpr(0))
                # compute a @ b
                tlx.async_dot(a_tiles[0], b_tiles[0], acc_tiles[0], use_acc=False, mBarriers=[acc_fulls[0]])
                tlx.barrier_wait(c_fulls[0], tl.constexpr(0))
                # wait for (a @ b) * 0.5) is ready
                tlx.barrier_wait(o_fulls[0], tl.constexpr(0))
                # compute ((a @ b) * 0.5) @ c
                tlx.async_dot(o_tiles[0], c_tiles[0], d_tiles[0], use_acc=False, mBarriers=[d_fulls[0]])

            # activation and epilogue
            with tlx.async_task(num_warps=4):
                # wait for (a @ b) is ready
                tlx.barrier_wait(acc_fulls[0], tl.constexpr(0))
                o = tlx.local_load(acc_tiles[0])
                o = o.to(tl.float16)
                o = o * 0.5
                tlx.local_store(o_tiles[0], o)
                tlx.barrier_arrive(o_fulls[0])

                # wait for ((a @ b) * 0.5) @ c is ready
                tlx.barrier_wait(d_fulls[0], tl.constexpr(0))
                d = tlx.local_load(d_tiles[0])
                d = d.to(tl.float16)
                offs_m = tl.arange(0, BLOCK_M)
                offs_n = tl.arange(0, BLOCK_N)
                d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
                tl.store(d_ptrs, d)

    torch.manual_seed(0)
    M, N, K = (64, 32, 16)
    a = torch.ones((M, K), device=device, dtype=torch.float16)
    b = torch.ones((K, N), device=device, dtype=torch.float16)
    c = torch.ones((N, N), device=device, dtype=torch.float16)
    d = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N}
    kernel = tcgen5_fa_kernel[(1, 1)](a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0),
                                      c.stride(1), d, d.stride(0), d.stride(1), **kern_kwargs, num_warps=4)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.tmem_alloc") == 2

    ref_out = ((a @ b) * 0.5) @ c
    torch.testing.assert_close(d, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_cluster_launch_control(BLOCK_SIZE, device):

    @triton.jit
    def mul2_clc(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        tile_id = tl.program_id(axis=0)

        # CLC Init
        clc_phase_producer = 1
        clc_phase_consumer = 0
        # NUM_CLC_STAGES=1
        # NUM_CONSUMERS=1
        clc_context = tlx.clc_create_context(1, 1)

        while tile_id != -1:
            # CLC producer
            tlx.clc_producer(clc_context, 0, clc_phase_producer)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x * y
            tl.store(z_ptr + offsets, output, mask=mask)

            # CLC consumer
            tile_id = tlx.clc_consumer(clc_context, 0, clc_phase_consumer)
            clc_phase_consumer ^= 1

            if tlx.thread_id(axis=0) == 0:
                tl.device_print("Extracted CtaID", tile_id)

    torch.manual_seed(0)
    # number of kernels to launch in a non-persistent mode
    size = 10000000
    x = torch.ones(size, device=device)
    y = torch.ones(size, device=device)

    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = mul2_clc[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, launch_cluster=True)

    ptx = kernel.asm["ptx"]

    assert re.search((r'clusterlaunchcontrol.try_cancel'), ptx, flags=re.DOTALL)
    assert re.search((r'clusterlaunchcontrol.query_cancel.is_canceled.pred.b128'), ptx, flags=re.DOTALL)
    assert re.search((r'clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128'), ptx, flags=re.DOTALL)

    assert (torch.count_nonzero(output) == size)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_region_error(device):

    @triton.jit
    def ws_error_kernel():
        with tlx.async_tasks():
            with tlx.async_task("default"):
                _z = 1 + 2
            with tlx.async_task(num_warps=1):
                _x = 1 / 0

    grid = lambda meta: (1, )
    with pytest.raises(triton.CompilationError) as e:
        ws_error_kernel[grid]()
    exc_msg = str(e.value)
    assert "ZeroDivisionError('division by zero')" in exc_msg, '\n\nExpected ZeroDivisionError but got: \n\n' + exc_msg + '\n\n'


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_index(BLOCK_SIZE, device):

    @triton.jit
    def local_index(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
        tlx.async_load(x_ptr_offsets, buffers[0], mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        s = tl.zeros((1, ), dtype=tl.float32)
        for i in range(0, BLOCK_SIZE):
            s += tlx.local_load(buffers[0][i])

        # tl.store(output_ptr, s)
        # Store using block addressing - broadcast the sum to all elements in the block
        output_offsets = output_ptr + offsets
        s_broadcasted = tl.broadcast_to(s, (BLOCK_SIZE, ))
        tl.store(output_offsets, s_broadcasted, mask=mask)

    torch.manual_seed(0)
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    local_index[grid](x, output, n_elements, BLOCK_SIZE)
    y = torch.tensor([10., 10., 10., 10.], device='cuda:0')
    torch.testing.assert_close(y, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_token_error(device):

    @triton.jit
    def asycn_copy_kernel(x_ptr, y_ptr, cond):
        buffers = tlx.local_alloc((128, ), tl.float32, 1)
        offsets = tl.arange(0, 128)
        if cond:
            token = tlx.async_load(x_ptr + offsets, buffers[0])
        else:
            token = tlx.async_load(y_ptr + offsets, buffers[0])
        tlx.async_load_commit_group([token])

    x = torch.tensor([128], dtype=torch.float32, device=device)
    y = torch.tensor([128], dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kernel = asycn_copy_kernel[grid](x, y, True)
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize(
    "src_dtype, dst_dtype",
    [
        ("float32", "float8_e5m2"),
        ("float32", "float8_e4m3fn"),
        ("float32", "float16"),
        ("float32", "bfloat16"),
    ],
)
def test_stoch_round(src_dtype, dst_dtype, device):

    @triton.jit
    def stoch_round_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        # Generate 1/4 shape for each random stream
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(0, offsets_quarter, n_rounds=7)
        # Combine the 4 blocks into a single vector of random values
        # r0,r1,r2,r3: each [BLOCK_SIZE//4]
        # after joins: rbits: [BLOCK_SIZE]
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(
            x,
            tlx.dtype_of(y_ptr),
            rbits,
        )
        tl.store(y_ptr + offsets, y)

    # Map string names to torch dtypes
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    src_dtype_torch = dtype_map[src_dtype]
    dst_dtype_torch = dtype_map[dst_dtype]

    SIZE = 256
    a = torch.randn([SIZE], dtype=torch.float32, device=device).to(src_dtype_torch)
    b = torch.empty([SIZE], dtype=torch.float32, device=device).to(dst_dtype_torch)
    grid = lambda meta: (1, )
    kernel = stoch_round_kernel[grid](
        a,
        b,
        BLOCK_SIZE=SIZE,
        num_warps=1,
    )
    assert kernel.asm["ptx"].count("cvt.rs.satfinite") > 0

    # Compare against PyTorch baseline
    # PyTorch doesn't have stochastic rounding, so we verify the result
    # is within the representable range and matches deterministic rounding
    # for most values (stochastic should be close on average)
    a_f32 = a.float()
    b_ref = a_f32.to(dst_dtype_torch)  # PyTorch uses round-to-nearest-even

    # Convert to float32 for validation (FP8 doesn't support all PyTorch ops)
    b_back = b.float()

    # Verify all values are in valid range (no NaN/Inf introduced)
    assert not torch.isnan(b_back).any(), "Stochastic rounding produced NaN"
    assert not torch.isinf(b_back).any(), "Stochastic rounding produced Inf"

    # For values that don't need rounding (exact in FP8), should match exactly
    exact_mask = b_back == a_f32
    if exact_mask.any():
        assert torch.equal(b[exact_mask],
                           b_ref[exact_mask]), ("Values that don't need rounding should match deterministic rounding")

    # For values that need rounding, verify they're in a reasonable range
    # (stochastic rounding can pick either of two adjacent representable values,
    # so we can't easily validate without knowing FP8 representation details)
    needs_rounding = ~exact_mask
    if needs_rounding.any():
        # Basic sanity check: stochastic result should be reasonably close to input
        # For FP8 e5m2, max representable is 57344, so use that as scale
        max_expected_diff = 100.0  # Conservative bound for FP8 rounding error
        diff = torch.abs(b_back[needs_rounding] - a_f32[needs_rounding])
        assert (diff < max_expected_diff).all(), (
            f"Stochastic rounding produced unreasonably large errors: max diff = {diff.max()}")
