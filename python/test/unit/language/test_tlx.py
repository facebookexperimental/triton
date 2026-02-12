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
    pattern_ws = r"ttg.warp_specialize(.*) attributes {requestedRegisters = array<i32: 120, 100, 100>}"
    assert re.search(pattern_ws, ttgir, flags=re.DOTALL)
    pattern_p0 = r"partition0\([^\n]*\)\s+num_warps\(4\)"
    assert re.search(pattern_p0, ttgir, flags=re.DOTALL)
    pattern_p1 = r"partition1\([^\n]*\)\s+num_warps\(1\)"
    assert re.search(pattern_p1, ttgir, flags=re.DOTALL)
    pattern_p2 = r"partition2\([^\n]*\)\s+num_warps\(1\)"
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
    pattern_cst = r"= arith.constant dense\<.*\>"
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

    output = torch.zeros(32, dtype=torch.int32, device="cuda")
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1, )](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device="cuda")
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
    def wgmma_kernel_A_smem(
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
    def wgmma_kernel_A_reg(
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
    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N}
    kernel = wgmma_kernel_A_smem[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0),
                                         z.stride(1), **kern_kwargs)
    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    z_ref = torch.matmul(x, y)
    torch.testing.assert_close(z, z_ref)

    # test reg
    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N}
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
    def tcgen5_dot_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
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

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}
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
    def tcgen5_dot_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr1,
        stride_cm,
        stride_cn,
        c_ptr2,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
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

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}
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
    def tcgen5_dot_kernel2cta_tma(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        A_TMEM: tl.constexpr,
    ):
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
        tlx.barrier_arrive(cta_bars[0], arrive_count=1, remote_cta_rank=0)
        tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

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
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "OUT_DTYPE": tl.float32,
        "M": M,
        "N": N,
        "K": K,
        "A_TMEM": A_TMEM,
    }
    kernel = tcgen5_dot_kernel2cta_tma[(M // BLOCK_M, N // BLOCK_N)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        ctas_per_cga=(2, 1, 1),  # TLX way: explicitly set cluster dims
        **kern_kwargs,
    )

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas to be 1 when using ctas_per_cga, got {kernel.metadata.num_ctas}")

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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper/Blackwell")
def test_cluster_dims(device):

    @triton.jit
    def test_kernel():
        pid = tl.program_id(axis=0)
        if pid == 0:
            return

    k = kernel = test_kernel[(2, )](ctas_per_cga=(2, 1, 1))
    assert kernel.metadata.cluster_dims == (2, 1, 1)
    assert ('"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32'
            in k.asm["ttgir"])


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper/Blackwell for DSM")
def test_remote_shmem_store(device):

    @triton.jit
    def remote_shmem_store_kernel(
        x,
        y,
    ):
        local_buff = tlx.local_alloc((1, ), tl.float32, 2)
        cluster_cta_rank = tlx.cluster_cta_rank()
        remote_store_view = tlx.local_view(local_buff, cluster_cta_rank ^ 1)
        offset = tl.arange(0, 1) + cluster_cta_rank
        value = tl.load(x + offset) + (cluster_cta_rank + 1) * 100

        tlx.remote_shmem_store(
            dst=remote_store_view,
            src=value,
            remote_cta_rank=cluster_cta_rank ^ 1,
        )
        tlx.cluster_barrier()
        local_load_view = tlx.local_view(local_buff, cluster_cta_rank)
        remote_value = tlx.local_load(local_load_view)
        tl.store(y + offset, remote_value)

    x = torch.empty((2, ), device=device, dtype=torch.float32)
    x[0] = 42.0
    x[1] = 43.0
    y = torch.empty((2, ), device=device, dtype=torch.float32)
    remote_shmem_store_kernel[(2, )](x, y, ctas_per_cga=(2, 1, 1))
    assert y[1] == 142.0 and y[0] == 243.0


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_async_remote_shmem_store(num_ctas, device):
    """Test that remote_shmem_store correctly aggregates 2D data across multiple CTAs."""

    @triton.jit
    def remote_store_sum_kernel(
        input_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NUM_CTAS: tl.constexpr,
    ):
        # Configure the number of CTAs participating in reduction
        BLOCK_N: tl.constexpr = triton.cdiv(N, NUM_CTAS)

        # Allocate NUM_CTAS buffers in shared memory, each with shape (BLOCK_M,)
        # to hold a 1D vector of float32 values
        local_buffs = tlx.local_alloc((BLOCK_M, ), tl.float32, NUM_CTAS)

        # Allocate barriers for synchronization across CTAs
        # Each non-zero CTA will use a barrier to signal when its data is written
        barriers = tlx.alloc_barriers(num_barriers=NUM_CTAS)

        # CTA 0 expects to receive (NUM_CTAS - 1) tiles from other CTAs
        # Each tile is BLOCK_M * sizeof(float32) bytes
        for i in tl.static_range(1, NUM_CTAS):
            tlx.barrier_expect_bytes(barriers[i], BLOCK_M * tlx.size_of(tl.float32))

        # Synchronize all CTAs before starting computation
        tlx.cluster_barrier()

        # Get the rank of this CTA within the cluster
        cta_rank = tlx.cluster_cta_rank()

        # Each CTA processes its portion of the input data (2D tile)
        # Layout: each CTA gets a different BLOCK_N columns
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = cta_rank * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load 2D tile: (BLOCK_M, BLOCK_N)
        offsets = offs_m[:, None] * N + offs_n[None, :]
        data = tl.load(input_ptr + offsets)

        # Compute sum over this tile along N dimension, resulting in shape [BLOCK_M]
        local_sum = tl.sum(data, axis=1)

        # Non-zero CTAs: send their 2D tile to CTA 0's shared memory asynchronously
        if cta_rank != 0:
            tlx.async_remote_shmem_store(dst=local_buffs[cta_rank],  # Destination buffer in CTA 0's shared memory
                                         src=local_sum,  # Source 2D tensor from this CTA
                                         remote_cta_rank=0,  # Target CTA is CTA 0
                                         barrier=barriers[cta_rank],  # Signal barrier when write completes
                                         )

        # CTA 0: aggregate all tiles and write final result
        if cta_rank == 0:
            # Start with CTA 0's own local sum
            final_sum = local_sum

            # Wait for each non-zero CTA to write its data, then accumulate
            for i in tl.static_range(1, NUM_CTAS):
                tlx.barrier_wait(barriers[i], phase=0)  # Wait for CTA i's data
                final_sum += tlx.local_load(local_buffs[i])  # Accumulate CTA i's sum

            # Write the final aggregated sum to output
            offs_m = tl.arange(0, BLOCK_M)
            tl.store(output_ptr + offs_m, final_sum)

    torch.manual_seed(0)
    M = 64
    N = 256
    input_tensor = torch.randn((M, N), dtype=torch.float32, device=device)
    output = torch.zeros(M, dtype=torch.float32, device=device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), META["NUM_CTAS"])

    kernel = remote_store_sum_kernel[grid](input_tensor, output, M=M, N=N, BLOCK_M=64, NUM_CTAS=num_ctas, num_warps=1,
                                           ctas_per_cga=(1, num_ctas, 1))

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.async_remote_shmem_store") == 1

    expected = torch.sum(input_tensor, dim=1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


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
    def tcgen5_dot_kernel2cta_tma_ws(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
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
                tlx.barrier_arrive(cta_bars[0], arrive_count=1, remote_cta_rank=0)
                tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

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
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "OUT_DTYPE": tl.float32,
        "M": M,
        "N": N,
        "K": K,
    }
    kernel = tcgen5_dot_kernel2cta_tma_ws[(M // BLOCK_M, N // BLOCK_N)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        ctas_per_cga=(2, 1, 1),
        **kern_kwargs,
    )

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}")

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


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_2cta(device):
    """
    Test 2-CTA scaled MMA generates tcgen05.mma.cta_group::2 instruction.
    Also verifies numerical correctness against reference implementation.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def tcgen5_dot_scaled_2cta_kernel(
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

        # difference from 1cta: B is split across 2 CTAs
        desc_b = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_K, BLOCK_N // 2],
        )

        desc_a_scale = tl.make_tensor_descriptor(
            a_scale_ptr,
            shape=[M // 128, K // 32 // 4, 2, 2 * 128],
            strides=[K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        # B scale is NOT split across CTAs - full scale needed for MMA
        desc_b_scale = tl.make_tensor_descriptor(
            b_scale_ptr,
            shape=[N // 128, K // 32 // 4, 2, 2 * 128],
            strides=[K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        # async load a and b into SMEM
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float8e4nv, tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tl.float8e4nv, tl.constexpr(1))  # difference from 1cta
        a_scale_tile = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tl.uint8, tl.constexpr(1))
        # B scale tile is NOT halved - full scale for MMA
        b_scale_tile = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tl.uint8, tl.constexpr(1))

        bars = tlx.alloc_barriers(tl.constexpr(4))
        bar_a = tlx.local_view(bars, 0)
        bar_b = tlx.local_view(bars, 1)
        bar_a_scale = tlx.local_view(bars, 2)
        bar_b_scale = tlx.local_view(bars, 3)
        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * 1)  # fp8
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * (BLOCK_N // 2) * 1)  # difference from 1cta: B is half
        tlx.barrier_expect_bytes(bar_a_scale, BLOCK_M // 128 * BLOCK_K // 32 // 4 * 2 * 2 * 128)
        tlx.barrier_expect_bytes(bar_b_scale, BLOCK_N // 128 * BLOCK_K // 32 // 4 * 2 * 2 * 128)  # full B scale

        # difference from 1cta: A offset by CTA rank, B offset by CTA rank
        tlx.async_descriptor_load(desc_a, a_tile[0], [cluster_cta_rank * BLOCK_M, 0], bar_a)
        tlx.async_descriptor_load(desc_b, b_tile[0], [0, cluster_cta_rank * BLOCK_N // 2], bar_b)
        tlx.async_descriptor_load(desc_a_scale, a_scale_tile[0], [cluster_cta_rank * BLOCK_M // 128, 0, 0, 0],
                                  bar_a_scale)
        tlx.async_descriptor_load(desc_b_scale, b_scale_tile[0], [0, 0, 0, 0], bar_b_scale)  # full B scale

        tlx.barrier_wait(bar_a, tl.constexpr(0))
        tlx.barrier_wait(bar_b, tl.constexpr(0))
        tlx.barrier_wait(bar_a_scale, tl.constexpr(0))
        tlx.barrier_wait(bar_b_scale, tl.constexpr(0))

        # difference from 1cta: CTA0 waits for both CTAs before issuing MMA op
        # "Arrive Remote, Wait Local" pattern: all CTAs signal CTA 0's barrier, only CTA 0 waits
        tlx.barrier_arrive(cta_bars[0], 1, remote_cta_rank=0)
        tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

        # Allocate barrier for MMA completion
        mma_done_bars = tlx.alloc_barriers(tl.constexpr(1))
        mma_done_bar = tlx.local_view(mma_done_bars, 0)

        # difference from 1cta: set two_ctas. Compiler auto generates pred to issue mma only from CTA0
        # Pass mma_done_bar directly to async_dot_scaled for MMA completion signaling
        tlx.async_dot_scaled(
            a_tile[0],
            b_tile[0],
            c_tile[0],
            a_scale_tile[0],
            A_format,
            b_scale_tile[0],
            B_format,
            use_acc=False,
            two_ctas=True,
            mBarriers=[mma_done_bar],
        )

        # Wait for MMA completion
        tlx.barrier_wait(mma_done_bar, tl.constexpr(0))

        result = tlx.local_load(c_tile[0])

        c = result.to(tl.float16)
        offs_m = cluster_cta_rank * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    # M=256 so BLOCK_M=128 per CTA, N=256 so BLOCK_N=256 total (128 per CTA for B data)
    M, N, K = (256, 256, 128)

    DTYPE_MAP = {
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(DTYPE_MAP[A_DATA_TYPE]).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(DTYPE_MAP[B_DATA_TYPE]).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    a_scale = torch.randint(10, 20, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(10, 20, (N, K // 32), dtype=torch.uint8, device=device)
    a_scale_4d = a_scale.reshape(a_scale.shape[0] // 128, a_scale.shape[1] // 4, 2, 2 * 128)
    b_scale_4d = b_scale.reshape(b_scale.shape[0] // 128, b_scale.shape[1] // 4, 2, 2 * 128)

    BLOCK_M = M // 2  # 128 per CTA
    BLOCK_N = N  # 256 total, 128 per CTA for B data
    BLOCK_K = K
    kern_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "M": M,
        "N": N,
        "K": K,
    }
    kernel = tcgen5_dot_scaled_2cta_kernel[(M // BLOCK_M, N // BLOCK_N)](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        a_scale_4d,
        b_scale_4d,
        c,
        c.stride(0),
        c.stride(1),
        A_DATA_TYPE,
        B_DATA_TYPE,
        ctas_per_cga=(2, 1, 1),  # TLX way: explicitly set cluster dims
        **kern_kwargs,
    )

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas to be 1 when using ctas_per_cga, got {kernel.metadata.num_ctas}")

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvgpu.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1
    assert ttgir.count("ttng.tc_gen5_mma_scaled") >= 1

    ptx = kernel.asm["ptx"]
    # The key assertion: with two_ctas=True, should generate cta_group::2 for scaled MMA
    assert ptx.count("tcgen05.mma.cta_group::2") > 0, (
        f"Expected tcgen05.mma.cta_group::2 for 2-CTA scaled MMA, but found: "
        f"cta_group::1 count={ptx.count('tcgen05.mma.cta_group::1')}, "
        f"cta_group::2 count={ptx.count('tcgen05.mma.cta_group::2')}")

    # Numeric verification: compute reference and compare
    def fp8e8m0_to_float32(scale):
        """Convert FP8 E8M0 scale values to float32."""
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference: D = (A * A_scale) @ (B * B_scale)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value 32 times along K dimension
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)

    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.parametrize("A_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.parametrize("B_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled(A_DATA_TYPE, B_DATA_TYPE, device):
    """
    Test D = (A * A_scale)  * (B * B_scale) with mxfp8 format for both A and B.

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements,
    matching cuBLAS block scaling layout.
    """

    VEC_SIZE = 32  # mxfp8 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tlx.dtype_of(a_scale_desc), tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tlx.dtype_of(b_scale_desc), tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 256)
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

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    # This matches cuBLAS block scaling layout used by tcgen5_mma_scaled
    a_scale = torch.randint(10, 20, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(10, 20, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Reshape to 5D format for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = a_scale.reshape(1, M // 128, K // VEC_SIZE // 4, 2, 2 * 128)
    b_scale_5d = b_scale.reshape(1, N // 128, K // VEC_SIZE // 4, 2, 2 * 128)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

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
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference
    a_scale_f32 = fp8e8m0_to_float32(a_scale_5d.reshape(M, K // VEC_SIZE))
    b_scale_f32 = fp8e8m0_to_float32(b_scale_5d.reshape(N, K // VEC_SIZE))
    # Repeats each scale value VEC_SIZE times along dimension 1.
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_tmem_scales(device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mxfp8 format and TMEM scales.

    This test verifies that scales can be stored in tensor memory (TMEM) instead
    of shared memory (SMEM). The scales are first loaded to SMEM via TMA, then
    copied to TMEM for use in the scaled MMA operation.
    """

    VEC_SIZE = 32  # mxfp8 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_tmem_scales_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = BLOCK_M // 128
        REP_N: tl.constexpr = BLOCK_N // 128
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K // 32, 4)

        # Allocate SMEM buffers for A, B, and scales
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        # 5D scale buffers in SMEM: [1, REP_M/N, REP_K, 2, 256]
        a_scale_smem = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tlx.dtype_of(a_scale_desc), tl.constexpr(1))
        b_scale_smem = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tlx.dtype_of(b_scale_desc), tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # Load scales to SMEM via TMA
        tlx.async_descriptor_load(a_scale_desc, a_scale_smem[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_smem[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Allocate TMEM for scales and accumulator
        # Scale shape in TMEM: flatten 5D to 2D for TMEM storage
        SCALE_K: tl.constexpr = BLOCK_K // 32
        SCALE_N: tl.constexpr = BLOCK_N // 32
        a_scale_tmem = tlx.local_alloc((BLOCK_M, SCALE_K), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_K, SCALE_N), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)

        # Copy scales from SMEM to TMEM directly using tmem_copy
        tlx.tmem_copy(a_scale_smem[0], a_scale_tmem[0])
        tlx.tmem_copy(b_scale_smem[0], b_scale_tmem[0])

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        # Use TMEM scales in async_dot_scaled
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[0], A_format, b_scale_tmem[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 256)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

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

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    a_scale = torch.randint(10, 20, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(10, 20, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Reshape to 5D format for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = a_scale.reshape(1, M // 128, K // VEC_SIZE // 4, 2, 2 * 128)
    b_scale_5d = b_scale.reshape(1, N // 128, K // VEC_SIZE // 4, 2, 2 * 128)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_tmem_scales_kernel[(1, 1)](
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
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1
    # Verify TMEM scales encoding is used
    assert "tensor_memory_scales_encoding" in ttgir
    # Verify tmem_copy is used for SMEM->TMEM transfer
    assert ttgir.count("ttng.tmem_copy") == 2

    # Converts E8M0 format scale values to float32
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference
    a_scale_f32 = fp8e8m0_to_float32(a_scale_5d.reshape(M, K // VEC_SIZE))
    b_scale_f32 = fp8e8m0_to_float32(b_scale_5d.reshape(N, K // VEC_SIZE))
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_tcgen05_commit(device):
    """
    Test tcgen05.commit tracking multiple tcgen05 ops
    """

    @triton.jit
    def tcgen5_commit_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr1,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        NUM_DOT: tl.constexpr,
    ):
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

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}

    num_dot = 4
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel = tcgen5_commit_kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z1,
        z1.stride(0),
        z1.stride(1),
        NUM_DOT=num_dot,
        **kern_kwargs,
    )
    ptx = kernel.asm["ptx"]
    assert ptx.count("tcgen05.mma") == 4 * num_dot  # loop unrolled so 4 mma ops per dot
    assert (ptx.count("tcgen05.commit") == 1 + num_dot
            )  # one for each dot (loop unrolled), then one dedicated barrier for all mma ops
    assert ptx.count("mbarrier.try_wait") == 2  # one for first sync dot, one for final wait
    ref_out = torch.zeros_like(z1)
    for _ in range(num_dot):
        ref_out += torch.matmul(x, y)
    torch.testing.assert_close(z1, ref_out)

    num_dot = 3
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel = tcgen5_commit_kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z1,
        z1.stride(0),
        z1.stride(1),
        NUM_DOT=num_dot,
        **kern_kwargs,
    )
    ptx = kernel.asm["ptx"]
    assert ptx.count("tcgen05.mma") == 4 * num_dot  # loop unrolled so 4 mma ops per dot
    assert (ptx.count("tcgen05.commit") == 1 + num_dot
            )  # one for each dot (loop unrolled), then one dedicated barrier for all mma ops
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
    def tcgen5_dot_kernel_tmem_A(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
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

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}
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
        assert ((ttgir.count("amdgpu.init_barrier") == 1) and (ttgir.count("amdgpu.read_barrier_phase") == 3)
                and (ttgir.count("amdgpu.arrive_barrier") == 3)), f"TTGIR {ttgir}"
    else:
        assert ((ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 3)
                and (ttgir.count("ttng.barrier_expect") == 0)
                and (ttgir.count("ttng.arrive_barrier") == 3)), f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_wait_arrive_ws(BLOCK_SIZE, device):
    kernel = run_tlx_square(tlx_square_ws, BLOCK_SIZE, device)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert ((ttgir.count("ttng.init_barrier") == 2) and (ttgir.count("ttng.wait_barrier") == 2)
            and (ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 2)
            and (ttgir.count("default {") == 1) and (ttgir.count("partition0") == 1)), f"TTGIR {ttgir}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_barrier_live_range(device):

    @triton.jit
    def bar_live_kernel():
        # an intentional early return here to check that we're considering dominance when inserting inval bar ops
        pid = tl.program_id(axis=0)
        if pid == 258:
            return

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
        tlx.fence_async_shared()
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
        tlx.fence_async_shared()
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

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.int16, tl.constexpr(1))
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
        tlx.fence_async_shared()
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
    assert "Please make sure that the type stays consistent" in "\n".join(list_msg)


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


def test_size_of(device):

    @triton.jit
    def size_of_kernel(output_ptr):
        # Test size_of for various dtypes
        size_fp32 = tlx.size_of(tl.float32)
        size_fp16 = tlx.size_of(tl.float16)
        size_int32 = tlx.size_of(tl.int32)
        size_int8 = tlx.size_of(tl.int8)
        size_int64 = tlx.size_of(tl.int64)

        # Store results
        tl.store(output_ptr + 0, size_fp32)
        tl.store(output_ptr + 1, size_fp16)
        tl.store(output_ptr + 2, size_int32)
        tl.store(output_ptr + 3, size_int8)
        tl.store(output_ptr + 4, size_int64)

    # Expected sizes in bytes
    expected_sizes = torch.tensor([4, 2, 4, 1, 8], dtype=torch.int32, device=device)
    output = torch.zeros(5, dtype=torch.int32, device=device)

    grid = lambda meta: (1, )
    size_of_kernel[grid](output)

    torch.testing.assert_close(output, expected_sizes)


def test_size_of_constexpr(device):

    @triton.jit
    def size_of_constexpr_kernel(output_ptr, DTYPE: tl.constexpr):
        # Test size_of with constexpr dtype argument
        size = tlx.size_of(DTYPE)
        tl.store(output_ptr, size)

    output = torch.zeros(1, dtype=torch.int32, device=device)

    # Test with float32 (4 bytes)
    grid = lambda meta: (1, )
    size_of_constexpr_kernel[grid](output, tl.float32)
    assert output.item() == 4, f"Expected 4 for float32, got {output.item()}"

    # Test with float16 (2 bytes)
    size_of_constexpr_kernel[grid](output, tl.float16)
    assert output.item() == 2, f"Expected 2 for float16, got {output.item()}"

    # Test with int8 (1 byte)
    size_of_constexpr_kernel[grid](output, tl.int8)
    assert output.item() == 1, f"Expected 1 for int8, got {output.item()}"

    # Test with int64 (8 bytes)
    size_of_constexpr_kernel[grid](output, tl.int64)
    assert output.item() == 8, f"Expected 8 for int64, got {output.item()}"


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dots_blackwell_tmem(device):
    """
    Test D = ((A@B) * 0.5) @ C
    """

    @triton.jit
    def tcgen5_fa_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        d_ptr,
        stride_dm,
        stride_dn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
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

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N}
    kernel = tcgen5_fa_kernel[(1, 1)](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        d,
        d.stride(0),
        d.stride(1),
        **kern_kwargs,
        num_warps=4,
    )

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
        clc_context = tlx.clc_create_context(1)

        while tile_id != -1:
            # CLC producer
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1

            block_start = tile_id * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x * y
            tl.store(z_ptr + offsets, output, mask=mask)

            # CLC consumer
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
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

    assert re.search((r"clusterlaunchcontrol.try_cancel"), ptx, flags=re.DOTALL)
    assert re.search((r"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128"), ptx, flags=re.DOTALL)
    assert re.search((r"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128"), ptx, flags=re.DOTALL)

    assert torch.count_nonzero(output) == size


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
    assert "ZeroDivisionError('division by zero')" in exc_msg, ("\n\nExpected ZeroDivisionError but got: \n\n" +
                                                                exc_msg + "\n\n")


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
    y = torch.tensor([10.0, 10.0, 10.0, 10.0], device="cuda:0")
    torch.testing.assert_close(y, output)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_mxfp4(device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mxfp4 (e2m1) format for both A and B.

    For mxfp4 format:
    - Two fp4 (e2m1) elements are packed into a single uint8
    - A has logical shape (M, K), packed along K to get physical shape (M, K//2)
    - B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    - B is transposed in SMEM before being passed to MMA to get (K//2, N)

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements,
    matching cuBLAS block scaling layout.
    """
    from triton.tools.mxfp import MXFP4Tensor

    VEC_SIZE = 32  # mxfp4 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_mxfp4_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        # A: (M, K//2) - packed along K
        # B: (N, K//2) - stored in transposed layout, packed along K
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_N, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8, tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8, tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K // 2 + BLOCK_N * BLOCK_K // 2
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Transpose B from (N, K//2) to (K//2, N) for MMA
        b_tile_T = tlx.local_trans(b_tile[0])

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile_T, c_tile[0], a_scale_tile[0], "e2m1", b_scale_tile[0], "e2m1",
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    # Create mxfp4 tensors and pack them
    # A has logical shape (M, K), packed along K to get physical shape (M, K//2)

    A = torch.full((M, K), 2, dtype=torch.float32, device=device)
    B = torch.full((N, K), 2, dtype=torch.float32, device=device)
    AMXFP4 = MXFP4Tensor(data=A, device=device)
    BMXFP4 = MXFP4Tensor(data=B, device=device)
    APACKED = AMXFP4.to_packed_tensor(dim=1)
    BPACKED = BMXFP4.to_packed_tensor(dim=1)

    a_ref = AMXFP4.to(torch.float32)

    # B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    # This matches the hardware expectation for mxfp4
    b_ref = BMXFP4.to(torch.float32).T  # Transpose for reference matmul -> (K, N)

    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    # TMA descriptors for packed mxfp4 data
    a_desc = TensorDescriptor.from_tensor(APACKED, [BLOCK_M, BLOCK_K // 2])
    b_desc = TensorDescriptor.from_tensor(BPACKED, [BLOCK_N, BLOCK_K // 2])  # B stored as (N, K//2)
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    # This matches cuBLAS block scaling layout used by tcgen5_mma_scaled
    a_scale = torch.randint(127, 128, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(127, 128, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Reshape to 5D format for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = a_scale.reshape(1, M // 128, K // VEC_SIZE // 4, 2, 2 * 128)
    b_scale_5d = b_scale.reshape(1, N // 128, K // VEC_SIZE // 4, 2, 2 * 128)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_mxfp4_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # print(a_scale_5d[0][0][0][0][0].item())
    a_scale_f32 = fp8e8m0_to_float32(a_scale_5d.reshape(M, K // VEC_SIZE))
    b_scale_f32 = fp8e8m0_to_float32(b_scale_5d.reshape(N, K // VEC_SIZE))
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32, b_ref * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "A_format,B_format",
    [("e4m3", "e2m1"),  # A is mxfp8, B is mxfp4
     ("e2m1", "e4m3"),  # A is mxfp4, B is mxfp8
     ],
)
@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_mixed_mxfp8_mxfp4(A_format, B_format, device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mixed mxfp8 (e4m3) and mxfp4 (e2m1) formats.

    This test exercises the fp4Padded logic in TLX's async_dot_scaled:
    - When A is mxfp4 and B is mxfp8: A_fp4Padded=True, B_fp4Padded=False
    - When A is mxfp8 and B is mxfp4: A_fp4Padded=False, B_fp4Padded=True

    For mxfp4 format:
    - Two fp4 (e2m1) elements are packed into a single uint8
    - Tensor is packed along K dimension, so shape (M, K) becomes (M, K//2)
    - B is stored transposed as (N, K//2) and transposed in SMEM to (K//2, N)

    For mxfp8 format:
    - Standard fp8 e4m3 layout with shape (M, K) or (K, N)

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements (cuBLAS block scaling layout).
    """
    from triton.tools.mxfp import MXFP4Tensor

    VEC_SIZE = 32  # mxfp uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_mixed_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_IS_FP4: tl.constexpr,
        B_IS_FP4: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        # For FP4: packed along K, so (M, K//2) or (N, K//2)
        # For FP8: full size (M, K) or (K, N)
        if A_IS_FP4:
            a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        else:
            a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))

        if B_IS_FP4:
            # B is stored transposed as (N, K//2) for FP4
            b_tile = tlx.local_alloc((BLOCK_N, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        else:
            # B is (K, N) for FP8
            b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))

        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256]
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8, tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8, tl.constexpr(1))

        # Calculate expected bytes for barrier
        if A_IS_FP4:
            A_BYTES: tl.constexpr = BLOCK_M * BLOCK_K // 2
        else:
            A_BYTES: tl.constexpr = BLOCK_M * BLOCK_K  # FP8 is 1 byte per element

        if B_IS_FP4:
            B_BYTES: tl.constexpr = BLOCK_N * BLOCK_K // 2
        else:
            B_BYTES: tl.constexpr = BLOCK_K * BLOCK_N  # FP8 is 1 byte per element

        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        tlx.barrier_expect_bytes(load_bar[0], A_BYTES + B_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Transpose B from (N, K//2) to (K//2, N) for FP4, or use as-is for FP8
        if B_IS_FP4:
            b_tile_for_mma = tlx.local_trans(b_tile[0])
        else:
            b_tile_for_mma = b_tile[0]

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile_for_mma, c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    A_IS_FP4 = A_format == "e2m1"
    B_IS_FP4 = B_format == "e2m1"

    # Create input tensors based on format
    if A_IS_FP4:
        # mxfp4: Create packed tensor (M, K//2)
        a_mxfp4 = MXFP4Tensor(data=torch.full((M, K), 2, dtype=torch.float32, device=device), device=device)
        a = a_mxfp4.to_packed_tensor(dim=1)  # Pack along K -> (M, K//2)
        a_ref = a_mxfp4.to(torch.float32)
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // 2])
    else:
        # mxfp8: Standard fp8 tensor (M, K)
        a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
        a_ref = a.to(torch.float32)
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])

    if B_IS_FP4:
        # mxfp4: Create packed tensor stored as (N, K//2), will be transposed in SMEM
        b_mxfp4 = MXFP4Tensor(data=torch.full((N, K), 2, dtype=torch.float32, device=device), device=device)
        b = b_mxfp4.to_packed_tensor(dim=1)  # Pack along K -> (N, K//2)
        b_ref = b_mxfp4.to(torch.float32).T  # Transpose for reference matmul -> (K, N)
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // 2])
    else:
        # mxfp8: Standard fp8 tensor (K, N)
        b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
        b_ref = b.to(torch.float32)
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])

    c = torch.zeros((M, N), device=device, dtype=torch.float16)
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    a_scale = torch.randint(127, 128, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(127, 128, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Reshape to 5D format for TMA (cuBLAS block scaling layout)
    a_scale_5d = a_scale.reshape(1, M // 128, K // VEC_SIZE // 4, 2, 2 * 128)
    b_scale_5d = b_scale.reshape(1, N // 128, K // VEC_SIZE // 4, 2, 2 * 128)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "A_IS_FP4": A_IS_FP4,
        "B_IS_FP4": B_IS_FP4,
    }
    kernel = tcgen5_dot_scaled_mixed_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format,
        B_format,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Check that fp4Padded is set correctly in the IR
    # When A is FP4 (mixed precision), A should have fp4Padded = true
    # When B is FP4 (mixed precision), B should have fp4Padded = true
    if A_IS_FP4:
        # First nvmma_shared (for A) should have fp4Padded = true
        assert "fp4Padded = true" in ttgir, "A should have fp4Padded=true when A is mxfp4 in mixed precision"
    if B_IS_FP4:
        # B's nvmma_shared should have fp4Padded = true
        assert "fp4Padded = true" in ttgir, "B should have fp4Padded=true when B is mxfp4 in mixed precision"

    # Converts E8M0 format scale values to float32
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference
    a_scale_f32 = fp8e8m0_to_float32(a_scale_5d.reshape(M, K // VEC_SIZE))
    b_scale_f32 = fp8e8m0_to_float32(b_scale_5d.reshape(N, K // VEC_SIZE))
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32, b_ref * b_scale_f32).to(torch.float16)

    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


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
            f"Stochastic rounding produced unreasonably large errors: max diff = {diff.max()}, "
            f"expected < {max_expected_diff}")


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("dst_dtype", ["float8_e5m2", "float8_e4m3fn", "float16", "bfloat16"])
def test_stoch_round_partial_pack(dst_dtype, device):
    """Test stochastic rounding with block sizes not evenly divisible by pack size."""

    @triton.jit
    def stoch_round_partial_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_ROUNDED: tl.constexpr,
                                   QUARTER_SIZE_ROUNDED: tl.constexpr):
        # Use power-of-2 size for arange (triton requirement), then mask to actual size
        offsets_full = tl.arange(0, BLOCK_SIZE_ROUNDED)
        mask = offsets_full < BLOCK_SIZE
        offsets = tl.where(mask, offsets_full, 0)
        x = tl.load(x_ptr + offsets, mask=mask)
        # For sizes that don't divide evenly by 4 (FP8 pack size)
        # Use pre-computed power-of-2 size for the quarter size
        offsets_quarter = tl.arange(0, QUARTER_SIZE_ROUNDED)
        r0, r1, r2, r3 = tl.randint4x(42, offsets_quarter, n_rounds=7)
        rbits_raw = tl.join(tl.join(r0, r1), tl.join(r2, r3))
        # Take only BLOCK_SIZE elements
        rbits = tl.view(rbits_raw, (BLOCK_SIZE_ROUNDED, ))
        rbits_masked = tl.where(mask, rbits, 0)
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits_masked)
        tl.store(y_ptr + offsets, y, mask=mask)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    dst_dtype_torch = dtype_map[dst_dtype]

    # Test with sizes not divisible by 4 (FP8) or 2 (BF16/F16)
    for SIZE in [130, 65, 17]:  # Not divisible by pack sizes
        # Round up SIZE to next power of 2
        SIZE_ROUNDED = 1 << (SIZE - 1).bit_length()
        # Compute quarter size and round it up to next power of 2
        quarter_size = (SIZE + 3) // 4
        QUARTER_SIZE_ROUNDED = 1 << (quarter_size - 1).bit_length()
        a = torch.randn([SIZE], dtype=torch.float32, device=device)
        b = torch.empty([SIZE], dtype=torch.float32, device=device).to(dst_dtype_torch)
        grid = lambda meta: (1, )
        stoch_round_partial_kernel[grid](
            a,
            b,
            BLOCK_SIZE=SIZE,
            BLOCK_SIZE_ROUNDED=SIZE_ROUNDED,
            QUARTER_SIZE_ROUNDED=QUARTER_SIZE_ROUNDED,
            num_warps=1,
        )

        # Verify no NaN/Inf
        b_back = b.float()
        assert not torch.isnan(b_back).any(), f"NaN produced for size {SIZE}"
        assert not torch.isinf(b_back).any(), f"Inf produced for size {SIZE}"


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("invalid_src, invalid_dst", [("float16", "float8_e5m2"), ("bfloat16", "float16"),
                                                      ("float32", "int32")])
def test_stoch_round_invalid_dtypes(invalid_src, invalid_dst, device):
    """Test that invalid dtype combinations raise proper errors."""

    @triton.jit
    def stoch_round_invalid_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, SRC_DTYPE: tl.constexpr,
                                   DST_DTYPE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(SRC_DTYPE)
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(0, offsets_quarter, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(x, DST_DTYPE, rbits)
        tl.store(y_ptr + offsets, y)

    dtype_map = {
        "float32": tl.float32,
        "float16": tl.float16,
        "bfloat16": tl.bfloat16,
        "float8_e5m2": tl.float8e5,
        "float8_e4m3fn": tl.float8e4nv,
        "int32": tl.int32,
    }

    SIZE = 128
    a = torch.randn([SIZE], dtype=torch.float32, device=device)
    b = torch.empty([SIZE], dtype=torch.float32, device=device)
    grid = lambda meta: (1, )

    with pytest.raises(Exception) as exc_info:
        stoch_round_invalid_kernel[grid](a, b, BLOCK_SIZE=SIZE, SRC_DTYPE=dtype_map[invalid_src],
                                         DST_DTYPE=dtype_map[invalid_dst], num_warps=1)

    # Verify error message mentions the issue
    error_msg = str(exc_info.value)
    assert "Stochastic rounding" in error_msg or "float32" in error_msg or "supported" in error_msg.lower()


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_stoch_round_entropy_quality(device):
    """Test that different random seeds produce different results."""

    @triton.jit
    def stoch_round_seed_kernel(x_ptr, y_ptr, seed, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        offsets_quarter = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, offsets_quarter, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits)
        tl.store(y_ptr + offsets, y)

    SIZE = 256
    # Use values that will definitely need rounding in FP8
    a = torch.randn([SIZE], dtype=torch.float32, device=device) * 10.0
    b1 = torch.empty([SIZE], dtype=torch.float8_e5m2, device=device)
    b2 = torch.empty([SIZE], dtype=torch.float8_e5m2, device=device)
    grid = lambda meta: (1, )

    # Run with different seeds
    stoch_round_seed_kernel[grid](a, b1, seed=12345, BLOCK_SIZE=SIZE, num_warps=1)
    stoch_round_seed_kernel[grid](a, b2, seed=67890, BLOCK_SIZE=SIZE, num_warps=1)

    # Results should be different for at least some values
    different_count = (b1.float() != b2.float()).sum().item()
    assert different_count > SIZE * 0.1, (
        f"Different seeds should produce different results, but only {different_count}/{SIZE} values differ")


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
    a_scale = torch.randint(10, 20, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(10, 20, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Reshape to 5D format for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = a_scale.reshape(1, M // 128, K // VEC_SIZE // 4, 2, 2 * 128)
    b_scale_5d = b_scale.reshape(1, N // 128, K // VEC_SIZE // 4, 2, 2 * 128)

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

    a_scale_f32 = fp8e8m0_to_float32(a_scale_5d.reshape(M, K // VEC_SIZE))
    b_scale_f32 = fp8e8m0_to_float32(b_scale_5d.reshape(N, K // VEC_SIZE))
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_buffer_indexing_in_function_call(device):
    """Test that buffer indexing with [] syntax works correctly in function calls"""

    @triton.jit
    def helper_function(buffers, idx, data):
        """Helper function that receives buffers and performs indexing inside"""
        tlx.local_store(buffers[idx], data)  # Indexing happens inside the helper
        result = tlx.local_load(buffers[idx])  # Indexing again
        return result

    @triton.jit
    def kernel_with_indexing(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Allocate buffer with multiple stages
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, num=tl.constexpr(4))

        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)

        # Pass buffers to helper function which performs ALL indexing
        result = helper_function(buffers, 0, x)

        # Store result
        tl.store(y_ptr + offsets, result, mask=mask)

    torch.manual_seed(0)
    size = 1024
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(size, BLOCK_SIZE), )
    kernel_with_indexing[grid](x, y, size, BLOCK_SIZE)

    # Verify correctness
    assert torch.allclose(y, x), "Buffer indexing in function call failed"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_async_tasks_warp_group_start_ids(BLOCK_SIZE, device):
    """Test that warp_group_start_id is correctly passed to warp_specialize op."""

    @triton.jit
    def warp_specialized_kernel_with_start_ids(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default"):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=2, warp_group_start_id=4, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                tl.store(z_ptr + offsets, x, mask=mask)
            with tlx.async_task(num_warps=1, warp_group_start_id=8):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                y = tl.load(y_ptr + offsets, mask=mask)
                tl.store(z_ptr + offsets, y, mask=mask)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = warp_specialized_kernel_with_start_ids[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE,
        num_warps=4,
    )
    ttgir = kernel.asm["ttgir"]

    # Verify that warpGroupStartIds attribute is present in the IR with the correct values
    pattern_ws = r"ttg.warp_specialize.*warpGroupStartIds = array<i32: 4, 6, 8>"
    assert re.search(pattern_ws, ttgir,
                     flags=re.DOTALL), (f"Expected warpGroupStartIds = array<i32: 4, 6, 8> in ttgir, got:\n{ttgir}")

    # Verify partition structure
    # Task 1 has replicate=2 with num_warps=2, so partition0 and partition1 both have 2 warps
    # Task 2 has replicate=1 with num_warps=1, so partition2 has 1 warp
    pattern_p0 = r"partition0\([^\n]*\)\s+num_warps\(2\)"
    assert re.search(pattern_p0, ttgir, flags=re.DOTALL)
    pattern_p1 = r"partition1\([^\n]*\)\s+num_warps\(2\)"
    assert re.search(pattern_p1, ttgir, flags=re.DOTALL)
    pattern_p2 = r"partition2\([^\n]*\)\s+num_warps\(1\)"
    assert re.search(pattern_p2, ttgir, flags=re.DOTALL)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer for cluster support")
def test_ctas_per_cga(device):
    """Test launching kernels with 2x1x1 ctas_per_cga (CUDA cluster dimensions) in autotune config."""

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE": 64},
                num_warps=4,
            ),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def simple_kernel_clustered(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(x_ptr + offsets, offsets, mask=mask)

    x = torch.zeros(256, dtype=torch.float32, device=device)
    num_blocks = triton.cdiv(256, 64)

    # Launch with autotuned config containing ctas_per_cga=(2,1,1)
    kernel = simple_kernel_clustered[(num_blocks, )](x, 256, ctas_per_cga=(2, 1, 1))

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}")


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell for TMEM")
def test_dummy_layout_function_inlining(device):
    """Test that dummy layouts are correctly resolved when helper functions are inlined into async tasks.

    This test verifies that:
    1. Helper functions with TMA+TMEM operations get properly inlined into async task regions
    2. The dummy layout resolution uses the correct num_warps from the async task context
       (not the global num_warps)
    3. TMA load/store and TMEM operations work correctly when in separate helper functions
       with different warp counts than the async task
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def load_helper(desc, smem_buffer, tmem_buffer, offset_m, offset_n, bar, tmem_full_bar):
        """Helper function: TMA load from global to SMEM, then store to TMEM."""
        tlx.async_descriptor_load(desc, smem_buffer, [offset_m, offset_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)
        # Load from SMEM to registers, then store to TMEM
        reg_data = tlx.local_load(smem_buffer)
        tlx.local_store(tmem_buffer, reg_data)
        # Signal that TMEM is ready
        tlx.barrier_arrive(tmem_full_bar)

    @triton.jit
    def store_helper(desc, smem_buffer, tmem_buffer, offset_m, offset_n, tmem_full_bar):
        """Helper function: Load from TMEM, then TMA store to global."""
        # Wait for TMEM to be ready
        tlx.barrier_wait(tmem_full_bar, phase=0)
        # Load from TMEM to registers, then store to SMEM
        reg_data = tlx.local_load(tmem_buffer)
        tlx.local_store(smem_buffer, reg_data)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(desc, smem_buffer, [offset_m, offset_n])
        tlx.async_descriptor_store_wait(0)

    @triton.jit
    def kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        # SMEM buffer for TMA operations
        smem_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, tl.constexpr(1))
        smem_buffer = tlx.local_view(smem_buffers, 0)

        # TMEM buffer for intermediate storage
        tmem_buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, tl.constexpr(1), tlx.storage_kind.tmem)
        tmem_buffer = tlx.local_view(tmem_buffers, 0)

        # Barrier for TMA load completion
        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_M * BLOCK_N * 2)

        # Barrier for TMEM write completion (producer-consumer sync between async tasks)
        tmem_full_bars = tlx.alloc_barriers(tl.constexpr(1))
        tmem_full_bar = tlx.local_view(tmem_full_bars, 0)

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        with tlx.async_tasks():
            with tlx.async_task("default"):
                # Load from TMA + store to TMEM
                load_helper(desc_in, smem_buffer, tmem_buffer, off_m, off_n, bar, tmem_full_bar)
            with tlx.async_task(num_warps=8):
                # Load from TMEM + store to TMA
                store_helper(desc_out, smem_buffer, tmem_buffer, off_m, off_n, tmem_full_bar)

    triton.set_allocator(alloc_fn)
    M, N = 128, 128
    BLOCK_M, BLOCK_N = 64, 64
    x = torch.randn((M, N), dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    compiled_kernel = kernel[grid](x, y, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)

    ttgir = compiled_kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 1
    assert ttgir.count("ttng.async_tma_copy_local_to_global") == 1
    assert ttgir.count("ttng.tmem_alloc") == 1
    assert ttgir.count("ttng.tmem_store") == 1
    assert ttgir.count("ttng.tmem_load") == 1

    assert torch.equal(x, y), "Data copy through TMA+TMEM should be exact"


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


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_barrier_wait_no_remote_view(device):
    """Test that barrier_wait does not allow remote_view of mbarrier."""

    @triton.jit
    def barrier_wait_remote_view_kernel():
        bars = tlx.alloc_barriers(num_barriers=tl.constexpr(1), arrive_count=1)
        bar = tlx.local_view(bars, 0)
        # Get remote view of the barrier
        remote_bar = tlx.remote_view(bar, 0)
        # This should raise an assertion error because barrier_wait does not support remote_view
        tlx.barrier_wait(remote_bar, phase=0)

    grid = lambda meta: (1, )
    with pytest.raises(triton.CompilationError) as e:
        barrier_wait_remote_view_kernel[grid](ctas_per_cga=(2, 1, 1))
    exc_msg = str(e.value)
    assert "barrier_wait" in exc_msg, f"Expected error about barrier_wait, but got: {exc_msg}"


@triton.jit
def _test_get_fp8_format_name_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
    EXPECTED: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    if result == EXPECTED:
        tl.store(output_ptr, 1)
    else:
        tl.store(output_ptr, 0)


@triton.jit
def _test_get_fp8_format_name_unsupported_kernel(
    output_ptr,
    DTYPE: tl.constexpr,
):
    result: tl.constexpr = tlx.get_fp8_format_name(DTYPE)
    tl.store(output_ptr, result == "e5m2")


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (tl.float8e5, "e5m2"),
        (tl.float8e4nv, "e4m3"),
    ],
)
def test_get_fp8_format_name(dtype, expected, device):
    """Test that FP8 dtypes return correct format strings."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    _test_get_fp8_format_name_kernel[(1, )](output, DTYPE=dtype, EXPECTED=expected)
    assert output.item() == 1


@pytest.mark.parametrize(
    "dtype",
    [
        tl.float32,
        tl.float16,
        tl.int32,
    ],
)
def test_get_fp8_format_name_unsupported_dtype_raises_error(dtype, device):
    """Test that non-FP8 dtypes raise a CompilationError during compilation."""
    output = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        _test_get_fp8_format_name_unsupported_kernel[(1, )](output, DTYPE=dtype)
    # Check that the underlying cause mentions the supported types
    assert "only supports tl.float8e5" in str(exc_info.value.__cause__)


class TestStorageKind:
    """Tests for tlx.storage_kind enum."""

    def test_storage_kind_values(self):
        assert tlx.storage_kind.smem.value == "smem"
        assert tlx.storage_kind.tmem.value == "tmem"
        assert tlx.storage_kind.smemCluster.value == "smemCluster"


class TestStorageAliasSpecType:
    """Tests for storage_alias_spec_type class."""

    def test_type_smem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes is None

    def test_type_tmem_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes is None

    def test_type_smem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        assert ty.storage == tlx.storage_kind.smem
        assert ty.buffer_size_bytes == 16384

    def test_type_tmem_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 32768)
        assert ty.storage == tlx.storage_kind.tmem
        assert ty.buffer_size_bytes == 32768

    def test_type_equality_same(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        assert ty1 == ty2

    def test_type_equality_different_storage(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 16384)
        assert ty1 != ty2

    def test_type_equality_different_size(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 32768)
        assert ty1 != ty2

    def test_type_equality_sized_vs_unsized(self):
        ty1 = tlx.storage_alias_spec_type(tlx.storage_kind.smem, 16384)
        ty2 = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert ty1 != ty2

    def test_type_repr_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        assert "smem" in repr(ty)
        assert "size" not in repr(ty)

    def test_type_repr_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 16384)
        assert "tmem" in repr(ty)
        assert "16384" in repr(ty)

    def test_type_mangle_unsized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.smem)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "smem" in mangle

    def test_type_mangle_sized(self):
        ty = tlx.storage_alias_spec_type(tlx.storage_kind.tmem, 8192)
        mangle = ty.mangle()
        assert "storage_alias_spec" in mangle
        assert "tmem" in mangle
        assert "8192" in mangle


class TestStorageAliasSpecClass:
    """Tests for the storage_alias_spec value class (not the builtin function)."""

    def test_class_smem_unsized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        assert buf.storage == tlx.storage_kind.smem
        assert buf.buffer_size_bytes is None
        assert buf.handle is None

    def test_class_tmem_sized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=32768,
        )
        assert buf.storage == tlx.storage_kind.tmem
        assert buf.buffer_size_bytes == 32768

    def test_class_rejects_smem_cluster(self):
        with pytest.raises(ValueError, match="smemCluster"):
            tlx.storage_alias_spec_type_class(
                handle=None,
                storage=tlx.storage_kind.smemCluster,
            )

    def test_class_type_attribute(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=4096,
        )
        assert isinstance(buf.type, tlx.storage_alias_spec_type)
        assert buf.type.storage == tlx.storage_kind.smem
        assert buf.type.buffer_size_bytes == 4096

    def test_class_immutability_storage(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        with pytest.raises(AttributeError):
            buf.storage = tlx.storage_kind.tmem

    def test_class_immutability_buffer_size(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
            buffer_size_bytes=1024,
        )
        with pytest.raises(AttributeError):
            buf.buffer_size_bytes = 2048

    def test_class_repr_unsized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        r = repr(buf)
        assert "storage_alias_spec" in r
        assert "smem" in r

    def test_class_repr_sized(self):
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=65536,
        )
        r = repr(buf)
        assert "storage_alias_spec" in r
        assert "tmem" in r
        assert "65536" in r


class TestLocalAllocWithStorageAliasSpec:
    """Tests for local_alloc accepting storage_alias_spec in reuse parameter."""

    def test_local_alloc_reuse_type_check_buffered_tensor(self):
        """Verify local_alloc accepts buffered_tensor in reuse (legacy behavior)."""
        # This is a type-level test - we can't fully test without a kernel context
        # but we verify the type annotation allows buffered_tensor
        import inspect
        from triton.language.extra.tlx.mem_ops import local_alloc as local_alloc_func

        sig = inspect.signature(local_alloc_func)
        reuse_param = sig.parameters["reuse"]
        # The annotation should include Union or | with both types
        annotation_str = str(reuse_param.annotation)
        assert "buffered_tensor" in annotation_str or "tlx.buffered_tensor" in annotation_str

    def test_local_alloc_reuse_type_check_storage_alias_spec(self):
        """Verify local_alloc accepts storage_alias_spec in reuse (new behavior)."""
        import inspect
        from triton.language.extra.tlx.mem_ops import local_alloc as local_alloc_func

        sig = inspect.signature(local_alloc_func)
        reuse_param = sig.parameters["reuse"]
        # The annotation should include Union or | with both types
        annotation_str = str(reuse_param.annotation)
        assert "storage_alias_spec" in annotation_str or "tlx.storage_alias_spec" in annotation_str

    def test_reuse_storage_mismatch_error_message(self):
        """Verify helpful error message when storage kinds don't match."""
        # Create a storage_alias_spec with smem storage
        buf = tlx.storage_alias_spec_type_class(
            handle=None,
            storage=tlx.storage_kind.smem,
        )
        # The error should mention both storage kinds when there's a mismatch
        # We can't fully test the error without a kernel context, but we can
        # verify the storage_alias_spec's storage property is accessible
        assert buf.storage == tlx.storage_kind.smem


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_thread_safety(device):
    """Verify that concurrent compilation of warp-specialized kernels is thread-safe.

    The TLX code generator uses thread-local storage for region_replica_id_stack
    and sub_region_has_exception. This test compiles two different kernels using
    async_tasks() + async_task_replica_id() from separate threads simultaneously
    to verify no cross-thread state corruption occurs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    @triton.jit
    def ws_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
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
                output = x + y + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, registers=100, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + y + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def ws_mul_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default", registers=120, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = a * b + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, registers=100, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = a * b + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    size = 98432
    BLOCK_SIZE = 1024

    def compile_and_run_add():
        torch.manual_seed(42)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_add_kernel[grid](x, y, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, x + y, check_dtype=False)
        return True

    def compile_and_run_mul():
        torch.manual_seed(43)
        a = torch.rand(size, device=device)
        b = torch.rand(size, device=device)
        out = torch.empty_like(a)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_mul_kernel[grid](a, b, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, a * b, check_dtype=False)
        return True

    # Use 4 workers: 2 run ws_add_kernel, 2 run ws_mul_kernel.
    # This tests both different-kernel and same-kernel concurrent compilation.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compile_and_run_add),
            executor.submit(compile_and_run_mul),
            executor.submit(compile_and_run_add),
            executor.submit(compile_and_run_mul),
        ]
        for future in as_completed(futures):
            assert future.result() is True


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_async_tasks_thread_exception_isolation(device):
    """Verify that a compilation exception in one thread doesn't affect others."""
    from concurrent.futures import ThreadPoolExecutor

    @triton.jit
    def ws_good_kernel(
        x_ptr,
        out_ptr,
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
                replica_id = tlx.async_task_replica_id()
                output = x + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=1, registers=100, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                replica_id = tlx.async_task_replica_id()
                output = x + replica_id - replica_id
                tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def ws_bad_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            # Missing "default" task — this should fail during compilation
            with tlx.async_task(num_warps=1, registers=100, replicate=2):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

    size = 98432
    BLOCK_SIZE = 1024

    def compile_and_run_good():
        x = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        ws_good_kernel[grid](x, out, n, BLOCK_SIZE, num_warps=4)
        torch.testing.assert_close(out, x, check_dtype=False)
        return True

    def compile_and_run_bad():
        x = torch.rand(size, device=device)
        out = torch.empty_like(x)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        try:
            ws_bad_kernel[grid](x, out, n, BLOCK_SIZE, num_warps=4)
        except Exception:
            pass  # Expected to fail
        return True

    # Run bad kernel first to set exception flag, then verify good kernel
    # still works on a thread that may be reused from the pool.
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit bad first, then good
        bad_future = executor.submit(compile_and_run_bad)
        bad_future.result()  # Wait for bad to finish
        good_future = executor.submit(compile_and_run_good)
        assert good_future.result() is True


class TestReuseGroupType:
    """Tests for tlx.reuse_group_type enum."""

    def test_reuse_group_type_values(self):
        assert tlx.reuse_group_type.shared.value == "shared"
        assert tlx.reuse_group_type.distinct.value == "distinct"

    def test_reuse_group_type_enum_members(self):
        # Verify all expected members exist
        members = list(tlx.reuse_group_type)
        assert len(members) == 2
        assert tlx.reuse_group_type.shared in members
        assert tlx.reuse_group_type.distinct in members


def _make_test_storage_alias_spec(storage: tlx.storage_kind = tlx.storage_kind.smem):
    """Helper to create a storage_alias_spec for testing reuse_group."""
    return tlx.storage_alias_spec_type_class(handle=None, storage=storage)


def _make_test_buffered_tensor(storage: tlx.storage_kind = tlx.storage_kind.smem):
    """Helper to create a buffered_tensor for testing reuse_group."""
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=2)
    return tlx.buffered_tensor(
        handle=None,
        element_ty=tl.float32,
        shape=[64, 64],
        num=2,
        storage=storage,
        layout=layout,
    )


class TestReuseGroup:
    """Tests for tlx.reuse_group class."""

    def test_reuse_group_basic_shared(self):
        """Test basic reuse_group creation with shared type."""
        elem1 = _make_test_buffered_tensor()
        elem2 = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem1,
            elem2,
            group_type=tlx.reuse_group_type.shared,
        )
        assert group.args == (elem1, elem2)
        assert group.group_type == tlx.reuse_group_type.shared

    def test_reuse_group_basic_distinct(self):
        """Test basic reuse_group creation with distinct type."""
        elem1 = _make_test_buffered_tensor()
        elem2 = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem1,
            elem2,
            group_type=tlx.reuse_group_type.distinct,
        )
        assert group.args == (elem1, elem2)
        assert group.group_type == tlx.reuse_group_type.distinct

    def test_reuse_group_single_element(self):
        """Test reuse_group with a single element."""
        elem = _make_test_buffered_tensor()
        group = tlx.reuse_group(
            elem,
            group_type=tlx.reuse_group_type.shared,
        )
        assert len(group.args) == 1
        assert group.args[0] is elem

    def test_reuse_group_multiple_elements(self):
        """Test reuse_group with more than 2 elements."""
        elems = tuple(_make_test_buffered_tensor() for _ in range(4))
        group = tlx.reuse_group(
            *elems,
            group_type=tlx.reuse_group_type.distinct,
        )
        assert group.args == elems
        assert len(group.args) == 4

    def test_reuse_group_nested(self):
        """Test nested reuse_group (Flash Attention pattern)."""
        # Inner group: distinct elements
        p = _make_test_buffered_tensor()
        alpha = _make_test_buffered_tensor()
        inner_group = tlx.reuse_group(
            p,
            alpha,
            group_type=tlx.reuse_group_type.distinct,
        )

        # Outer group: shared with inner group
        qk = _make_test_buffered_tensor()
        outer_group = tlx.reuse_group(
            qk,
            inner_group,
            group_type=tlx.reuse_group_type.shared,
        )

        assert outer_group.group_type == tlx.reuse_group_type.shared
        assert len(outer_group.args) == 2
        assert outer_group.args[0] is qk
        assert outer_group.args[1] is inner_group
        assert inner_group.group_type == tlx.reuse_group_type.distinct

    def test_reuse_group_deeply_nested(self):
        """Test 3-level nested reuse_group."""
        # Level 3 (innermost)
        c = _make_test_buffered_tensor()
        d = _make_test_buffered_tensor()
        inner = tlx.reuse_group(
            c,
            d,
            group_type=tlx.reuse_group_type.shared,
        )

        # Level 2
        b = _make_test_buffered_tensor()
        middle = tlx.reuse_group(
            b,
            inner,
            group_type=tlx.reuse_group_type.distinct,
        )

        # Level 1 (outermost)
        a = _make_test_buffered_tensor()
        outer = tlx.reuse_group(
            a,
            middle,
            group_type=tlx.reuse_group_type.shared,
        )

        assert outer.group_type == tlx.reuse_group_type.shared
        assert outer.args[1].group_type == tlx.reuse_group_type.distinct
        assert outer.args[1].args[1].group_type == tlx.reuse_group_type.shared

    def test_reuse_group_empty_args_raises_error(self):
        """Test reuse_group raises error with empty args tuple."""
        with pytest.raises(ValueError, match="at least one element"):
            tlx.reuse_group(group_type=tlx.reuse_group_type.shared, )

    def test_reuse_group_nested_same_type_shared_raises_error(self):
        """Test that nesting shared inside shared raises an error."""
        elem = _make_test_buffered_tensor()
        inner = tlx.reuse_group(
            elem,
            group_type=tlx.reuse_group_type.shared,
        )
        with pytest.raises(ValueError, match="same group_type"):
            tlx.reuse_group(
                inner,
                group_type=tlx.reuse_group_type.shared,
            )

    def test_reuse_group_nested_same_type_distinct_raises_error(self):
        """Test that nesting distinct inside distinct raises an error."""
        elem = _make_test_buffered_tensor()
        inner = tlx.reuse_group(
            elem,
            group_type=tlx.reuse_group_type.distinct,
        )
        with pytest.raises(ValueError, match="same group_type"):
            tlx.reuse_group(
                inner,
                group_type=tlx.reuse_group_type.distinct,
            )

    def test_reuse_group_invalid_element_type_raises_error(self):
        """Test that invalid element types raise TypeError."""
        with pytest.raises(TypeError, match="must be buffered_tensor or reuse_group"):
            tlx.reuse_group(
                "invalid",
                group_type=tlx.reuse_group_type.shared,
            )

class TestToMxfp8:
    """Tests for the _to_mxfp8_block library function callable from JIT code with VEC_SIZE=32."""

    @staticmethod
    def _reference_mxfp8_quantize(data, vec_size, torch_dtype):
        """Python reference for MXFP8 quantization matching _compute_scale_and_quantize.

        Returns:
            scale_e8m0: uint8 tensor [M, K // vec_size]
            data_fp8: fp8 tensor [M, K]
        """
        fp8_max = torch.finfo(torch_dtype).max
        M, K = data.shape
        num_scales = K // vec_size
        data_f32 = data.float()
        data_reshaped = data_f32.reshape(M, num_scales, vec_size)
        max_abs = data_reshaped.abs().amax(dim=2)
        descale = max_abs / fp8_max
        log2_descale = torch.log2(descale)
        ceil_log2 = torch.ceil(log2_descale)
        clamped_exp = torch.clamp(ceil_log2, -127.0, 127.0)
        is_zero = descale < 1e-38
        biased_exp = torch.where(is_zero, torch.zeros_like(clamped_exp), clamped_exp + 127)
        scale_e8m0 = biased_exp.to(torch.uint8)
        descale_fp = torch.where(
            biased_exp == 0,
            torch.ones_like(biased_exp),
            torch.exp2(127 - biased_exp),
        )
        scaled_data = data_reshaped * descale_fp.unsqueeze(2)
        scaled_data = torch.clamp(scaled_data, -fp8_max, fp8_max)
        data_flat = scaled_data.reshape(M, K)
        data_fp8 = data_flat.to(torch_dtype)
        return scale_e8m0, data_fp8

    @staticmethod
    def _reference_swizzle_scales(scales):
        """Swizzle [128, 4] uint8 -> [512] uint8 matching _online_swizzle_prefill."""
        return scales.reshape(4, 32, 4).permute(1, 0, 2).reshape(512).contiguous()

    @staticmethod
    def _run_to_mxfp8_block(input_data, elem_dtype, device):
        """Run _to_mxfp8_block in a JIT kernel and return FP8 data and swizzled scales."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC_SIZE = 128, 128, 32

        @triton.jit
        def kernel(
            input_ptr,
            data_out_ptr,
            scale_out_ptr,
            BLOCK_M: tl.constexpr,
            BLOCK_K: tl.constexpr,
            VEC_SIZE: tl.constexpr,
            ELEM_DTYPE: tl.constexpr,
        ):
            offs_m = tl.arange(0, BLOCK_M)
            offs_k = tl.arange(0, BLOCK_K)
            data = tl.load(input_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
            if ELEM_DTYPE == "e4m3":
                fp8_type: tl.constexpr = tl.float8e4nv
            else:
                fp8_type: tl.constexpr = tl.float8e5
            data_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), fp8_type, tl.constexpr(1))
            scale_tile = tlx.local_alloc((1, 1, 1, 2, 256), tl.uint8, tl.constexpr(1))
            tlx._to_mxfp8_block(data, data_tile[0], scale_tile[0], VEC_SIZE, fp8_type)
            data_fp8 = tlx.local_load(data_tile[0])
            tl.store(data_out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], data_fp8)
            scale_loaded = tlx.local_load(scale_tile[0])
            scale_flat = tl.reshape(scale_loaded, [512])
            tl.store(scale_out_ptr + tl.arange(0, 512), scale_flat)

        data_out = torch.empty(M, K, dtype=torch_dtype, device=device)
        scale_out = torch.empty(512, dtype=torch.uint8, device=device)
        kernel[(1, )](input_data, data_out, scale_out, M, K, VEC_SIZE, elem_dtype)
        return data_out, scale_out

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_uniform(self, elem_dtype, device):
        """Test _to_mxfp8_block with uniform 1.0 input and VEC_SIZE=32."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC = 128, 128, 32
        input_data = torch.ones(M, K, dtype=torch.float32, device=device)

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        ref_scale, ref_data = self._reference_mxfp8_quantize(input_data.cpu(), VEC, torch_dtype)
        ref_scale_swizzled = self._reference_swizzle_scales(ref_scale)
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale_swizzled)

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_zeros(self, elem_dtype, device):
        """Test _to_mxfp8_block with all-zero input."""
        M, K = 128, 128
        input_data = torch.zeros(M, K, dtype=torch.float32, device=device)

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        assert torch.all(data_out.float() == 0)
        assert torch.all(scale_out == 0)

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_random(self, elem_dtype, device):
        """Test _to_mxfp8_block with random data against Python reference."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC = 128, 128, 32
        torch.manual_seed(42)
        input_data = torch.randn(M, K, dtype=torch.float32, device=device) * 100

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        ref_scale, ref_data = self._reference_mxfp8_quantize(input_data.cpu(), VEC, torch_dtype)
        ref_scale_swizzled = self._reference_swizzle_scales(ref_scale)
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale_swizzled)


class TestSetBufferOverlap:
    """Tests for tlx.set_buffer_overlap and storage_alias_spec.set_buffer_overlap method."""

    def test_set_buffer_overlap_compiles_to_ir(self):
        """Test that set_buffer_overlap compiles to IR but fails during lowering.

        This test verifies that the JIT API works correctly - the code should
        successfully lower to MLIR IR but is expected to fail in later lowering
        passes because offset calculation is not yet implemented.
        """

        @triton.jit
        def set_buffer_overlap_kernel(BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers using the spec
            a = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            b = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.bfloat16, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)

            # Define overlap scheme: a and b share the same memory region
            spec.set_buffer_overlap(tlx.reuse_group(a, b, group_type=tlx.reuse_group_type.shared))

        grid = lambda meta: (1, )

        # The kernel should compile to IR but fail during lowering
        # (offset calculation not implemented yet)
        with pytest.raises(RuntimeError):
            set_buffer_overlap_kernel[grid](BLOCK_SIZE=64)

    def test_set_buffer_overlap_nested_compiles_to_ir(self):
        """Test that nested reuse_group in set_buffer_overlap compiles to IR.

        This test verifies Flash Attention-style nested overlap schemes work
        at the JIT level.
        """

        @triton.jit
        def set_buffer_overlap_nested_kernel(BLOCK_SIZE: tl.constexpr):
            # Create a storage alias spec
            spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

            # Allocate buffers (Flash Attention pattern)
            qk = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                 reuse=spec)
            p = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.smem,
                                reuse=spec)
            alpha = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(2), tlx.storage_kind.smem, reuse=spec)

            # Define overlap scheme: QK shares with (P distinct from alpha)
            spec.set_buffer_overlap(
                tlx.reuse_group(
                    qk,
                    tlx.reuse_group(p, alpha, group_type=tlx.reuse_group_type.distinct),
                    group_type=tlx.reuse_group_type.shared,
                ))

        grid = lambda meta: (1, )

        # The kernel should compile to IR but fail during lowering
        with pytest.raises(RuntimeError):
            set_buffer_overlap_nested_kernel[grid](BLOCK_SIZE=64)

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_vote_ballot_sync(device):
    """Test vote_ballot_sync TLX operation for warp-level voting."""

    @triton.jit
    def vote_ballot_kernel(
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each thread's lane ID (use x-axis thread ID)
        tid = tlx.thread_id(0)

        # Create a predicate: lanes 0-15 vote True, lanes 16-31 vote False
        pred = tid < 16

        # Perform warp-level ballot vote
        # 0xFFFFFFFF means all 32 threads in the warp participate
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)

        # Store the ballot result from thread 0 only
        if tid == 0:
            tl.store(output_ptr, ballot_result)

    output = torch.zeros(1, dtype=torch.int32, device=device)

    # Run the kernel with 1 warp
    vote_ballot_kernel[(1, )](output, BLOCK_SIZE=32, num_warps=1)
    torch.cuda.synchronize()

    # Expected ballot result: threads 0-15 have pred=True, threads 16-31 have pred=False
    # So ballot should be 0x0000FFFF (lower 16 bits set)
    expected_ballot = 0x0000FFFF
    assert output.item() == expected_ballot, f"Expected {hex(expected_ballot)}, got {hex(output.item())}"


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_vote_ballot_sync_ir_emission(device):
    """Test that vote_ballot_sync generates the correct IR."""

    @triton.jit
    def vote_ballot_ir_kernel(output_ptr, ):
        tid = tlx.thread_id(0)
        pred = tid < 16  # First 16 threads True
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
        if tid == 0:
            tl.store(output_ptr, ballot_result)

    output = torch.zeros(1, dtype=torch.int32, device=device)
    kernel = vote_ballot_ir_kernel[(1, )](output, num_warps=1)

    # Verify the TTGIR contains the vote_ballot_sync op
    ttgir = kernel.asm["ttgir"]
    assert "vote_ballot_sync" in ttgir, "Expected vote_ballot_sync in TTGIR"

    # Verify the LLVM IR contains the NVVM vote instruction
    llir = kernel.asm["llir"]
    assert "nvvm.vote.ballot.sync" in llir or "vote.sync.ballot" in llir, (
        "Expected nvvm.vote.ballot.sync or vote.sync.ballot in LLVM IR")
