import math
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_hopper
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

from triton.runtime.fbcode_gating import is_fbcode_dependant

if is_fbcode_dependant():
    from python.test.unit.language.conftest import _generate_test_params, _swizzle_scale_to_5d
else:
    from conftest import _generate_test_params, _swizzle_scale_to_5d


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


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
@pytest.mark.parametrize("BLOCK", [64, 128])
def test_async_dot_local_store(BLOCK, device):
    """Test WGMMA dot result stored to SMEM via local_store then TMA-stored out."""

    @triton.jit
    def _kernel(desc_a, desc_b, desc_c, BLOCK: tl.constexpr):
        a_tiles = tlx.local_alloc((BLOCK, BLOCK), tlx.dtype_of(desc_a), 1)
        b_tiles = tlx.local_alloc((BLOCK, BLOCK), tlx.dtype_of(desc_b), 1)
        out_tiles = tlx.local_alloc((BLOCK, BLOCK), tlx.dtype_of(desc_c), 1)
        a_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=tl.constexpr(1))
        b_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=tl.constexpr(1))

        a_full = tlx.local_view(a_fulls, 0)
        tlx.barrier_expect_bytes(a_full, 2 * BLOCK * BLOCK)
        tlx.async_descriptor_load(desc_a, a_tiles, [0, 0], a_full)
        b_full = tlx.local_view(b_fulls, 0)
        tlx.barrier_expect_bytes(b_full, 2 * BLOCK * BLOCK)
        tlx.async_descriptor_load(desc_b, b_tiles, [0, 0], b_full)

        tlx.barrier_wait(a_full, 0)
        tlx.barrier_wait(b_full, 0)
        a_view = tlx.local_view(a_tiles, 0)
        b_view = tlx.local_view(b_tiles, 0)
        acc = tlx.async_dot(a_view, b_view)
        acc = tlx.async_dot_wait(0, acc)

        acc_fp16 = acc.to(tlx.dtype_of(desc_c))
        out_view = tlx.local_view(out_tiles, 0)
        tlx.local_store(out_view, acc_fp16)
        tlx.fence_async_shared()
        tlx.async_descriptor_store(desc_c, out_view, [0, 0])
        tlx.async_descriptor_store_wait(0)

    a = torch.randn(BLOCK, BLOCK, device=device, dtype=torch.float16)
    b = torch.randn(BLOCK, BLOCK, device=device, dtype=torch.float16)
    c = torch.empty(BLOCK, BLOCK, device=device, dtype=torch.float16)
    desc_a = TensorDescriptor(a, shape=[BLOCK, BLOCK], strides=[BLOCK, 1], block_shape=[BLOCK, BLOCK])
    desc_b = TensorDescriptor(b, shape=[BLOCK, BLOCK], strides=[BLOCK, 1], block_shape=[BLOCK, BLOCK])
    desc_c = TensorDescriptor(c, shape=[BLOCK, BLOCK], strides=[BLOCK, 1], block_shape=[BLOCK, BLOCK])

    _kernel[(1, )](desc_a, desc_b, desc_c, BLOCK=BLOCK, num_stages=0, num_warps=4)
    z_ref = torch.matmul(a, b)
    torch.testing.assert_close(c, z_ref)


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
@pytest.mark.parametrize("A_TMEM", [False, True])
@pytest.mark.parametrize("SAMPLE_M", [256, 128])
def test_async_dot_blackwell_2cta_tma(device, A_TMEM, SAMPLE_M):
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
        mma_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

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
            tlx.async_dot(buf_alloc_a_tmem[0], b_smem, acc_tmem, use_acc=False, mBarriers=[mma_bars[0]], two_ctas=True,
                          out_dtype=OUT_DTYPE)
        else:
            tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[mma_bars[0]], two_ctas=True,
                          out_dtype=OUT_DTYPE)
        tlx.barrier_wait(mma_bars[0], 0)
        result = tlx.local_load(acc_tmem)

        c = result.to(tl.float16)
        offs_m = cluster_cta_rank * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    M, N, K = (SAMPLE_M, 128, 128)
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
    assert kernel.metadata.ctas_per_cga == (2, 1, 1), (
        f"expecting ctas_per_cga to be (2, 1, 1), got {kernel.metadata.ctas_per_cga}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas to be 1 when using ctas_per_cga, got {kernel.metadata.num_ctas}")

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvg.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1

    ptx = kernel.asm["ptx"]
    assert ptx.count("fence.mbarrier_init.release.cluster") == 1
    assert ptx.count("fence.proxy.async.shared::cluster") >= 1
    # Verify ordering: fences → cluster sync → tmem alloc
    fence_mbar_pos = ptx.index("fence.mbarrier_init.release.cluster")
    fence_proxy_pos = ptx.index("fence.proxy.async.shared::cluster")
    cluster_arrive_pos = ptx.index("barrier.cluster.arrive.aligned")
    cluster_wait_pos = ptx.index("barrier.cluster.wait.aligned")
    tmem_alloc_pos = ptx.index("tcgen05.alloc.cta_group::2")
    assert fence_mbar_pos < fence_proxy_pos < cluster_arrive_pos < cluster_wait_pos < tmem_alloc_pos
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
                tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[tmem_full_bars[0]], two_ctas=True,
                              out_dtype=OUT_DTYPE)

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
    assert kernel.metadata.ctas_per_cga == (2, 1, 1), (
        f"expecting ctas_per_cga to be (2, 1, 1), got {kernel.metadata.ctas_per_cga}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas (not used in tlx) to be 1 but got {kernel.metadata.num_ctas}")

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvg.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1

    ptx = kernel.asm["ptx"]
    # Verify cluster sync and tmem alloc ordering in PTX:
    #   fence.mbarrier_init.release.cluster
    #   fence.proxy.async.shared::cluster
    #   barrier.cluster.arrive.aligned  (default side)
    #   barrier.cluster.wait.aligned
    #   tcgen05.alloc.cta_group::2      (tmem alloc after cluster sync)
    fence_mbar_pos = ptx.index("fence.mbarrier_init.release.cluster")
    fence_proxy_pos = ptx.index("fence.proxy.async.shared::cluster")
    cluster_arrive_pos = ptx.index("barrier.cluster.arrive.aligned", fence_proxy_pos)
    cluster_wait_pos = ptx.index("barrier.cluster.wait.aligned")
    tmem_alloc_pos = ptx.index("tcgen05.alloc.cta_group::2")
    assert fence_mbar_pos < fence_proxy_pos < cluster_arrive_pos < cluster_wait_pos < tmem_alloc_pos
    assert ptx.count("barrier.cluster.arrive.aligned") == 2
    assert ptx.count("barrier.cluster.wait.aligned") == 1
    assert ptx.count("mapa.shared::cluster") == 1  # address mapping for remote_view
    assert ptx.count("tcgen05.mma.cta_group::2") == 8  # BK=128 divided into steps of 16

    ref_out = torch.matmul(x, y)
    torch.testing.assert_close(z, ref_out)


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

    a_scale = torch.randint(124, 130, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // 32), dtype=torch.uint8, device=device)
    a_scale_4d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // 32), M // 128, K // 32 // 4).squeeze(0)
    b_scale_4d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // 32), N // 128, K // 32 // 4).squeeze(0)

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
    assert kernel.metadata.ctas_per_cga == (2, 1, 1), (
        f"expecting ctas_per_cga to be (2, 1, 1), got {kernel.metadata.ctas_per_cga}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas to be 1 when using ctas_per_cga, got {kernel.metadata.num_ctas}")

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvg.cluster_id") == 1
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
    a_scale = torch.randint(124, 130, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

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

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeats each scale value VEC_SIZE times along dimension 1.
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_tlx_tmem_copy_fp8_data_ttgir(device):

    @triton.jit
    def tmem_copy_fp8_data_kernel(a, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offsets = offs_m[:, None] * BLOCK_N + offs_n[None, :]

        a_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(a), tl.constexpr(1))
        a_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(a), tl.constexpr(1), tlx.storage_kind.tmem)

        a_reg = tl.load(a + offsets)
        tlx.local_store(a_smem[0], a_reg)
        tlx.tmem_copy(a_smem[0], a_tmem[0])

    BLOCK_M, BLOCK_N = 128, 128
    a = torch.empty((BLOCK_M, BLOCK_N), device=device, dtype=torch.float8_e4m3fn)

    kernel = tmem_copy_fp8_data_kernel[(1, )](a, BLOCK_M, BLOCK_N)

    ttgir = kernel.asm["ttgir"]
    assert "ttng.tmem_copy" in ttgir
    assert "tensor_memory_encoding" in ttgir
    assert "tensor_memory_scales_encoding" not in ttgir


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
    a_scale = torch.randint(124, 130, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

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

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_tmem_buffer_scales_two_entries(device):
    """
    Test storing to a TMEM buffer for scales with 2 entries.
    Stores all 0s (uint8) to entry 0 and all 127s (uint8) to entry 1,
    then verifies correctness by using each entry as scales in a
    separate scaled MMA operation.

    In E8M0 encoding, byte 0 maps to float 0.0 (so MMA result is zero)
    and byte 127 maps to 2^(127-127) = 1.0 (so MMA result equals the
    unscaled matmul).
    """

    @triton.jit
    def kernel(
        a_desc,
        b_desc,
        c0_desc,
        c1_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        SCALE_K: tl.constexpr = BLOCK_K // 32
        SCALE_N: tl.constexpr = BLOCK_N // 32

        # Load A, B to SMEM via TMA
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Allocate TMEM scale buffers with 2 entries
        a_scale_tmem = tlx.local_alloc((BLOCK_M, SCALE_K), tl.uint8, tl.constexpr(2), tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_K, SCALE_N), tl.uint8, tl.constexpr(2), tlx.storage_kind.tmem)

        # Entry 0: store all 0s
        tlx.local_store(a_scale_tmem[0], tl.full((BLOCK_M, SCALE_K), 0, tl.uint8))
        tlx.local_store(b_scale_tmem[0], tl.full((BLOCK_K, SCALE_N), 0, tl.uint8))

        # Entry 1: store all 127s
        tlx.local_store(a_scale_tmem[1], tl.full((BLOCK_M, SCALE_K), 127, tl.uint8))
        tlx.local_store(b_scale_tmem[1], tl.full((BLOCK_K, SCALE_N), 127, tl.uint8))

        # Accumulator in TMEM
        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

        # MMA with entry 0 scales
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[0], A_format, b_scale_tmem[0], B_format,
                             use_acc=False)
        result0 = tlx.local_load(c_tile[0])
        c0_desc.store([0, 0], result0.to(tlx.dtype_of(c0_desc)))

        # MMA with entry 1 scales
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[1], A_format, b_scale_tmem[1], B_format,
                             use_acc=False)
        result1 = tlx.local_load(c_tile[0])
        c1_desc.store([0, 0], result1.to(tlx.dtype_of(c1_desc)))

    torch.manual_seed(0)
    M, N, K = 128, 128, 256
    BLOCK_M, BLOCK_N, BLOCK_K = M, N, K

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    c0 = torch.zeros((M, N), device=device, dtype=torch.float16)
    c1 = torch.zeros((M, N), device=device, dtype=torch.float16)

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c0_desc = TensorDescriptor.from_tensor(c0, block_shape=[BLOCK_M, BLOCK_N])
    c1_desc = TensorDescriptor.from_tensor(c1, block_shape=[BLOCK_M, BLOCK_N])

    kernel[(1, 1)](
        a_desc,
        b_desc,
        c0_desc,
        c1_desc,
        A_DATA_TYPE,
        B_DATA_TYPE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    VEC_SIZE = 32

    # E8M0 byte 0 → float 0.0, so result is exactly 0
    torch.testing.assert_close(c0, torch.zeros_like(c0), atol=0, rtol=0)

    # E8M0 byte 127 → float 2^(127-127) = 1.0, so result equals unscaled matmul
    ref_c1 = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(c1, ref_c1, atol=atol, rtol=0)


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

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

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

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
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

    # Swizzle to 5D cuBLAS block scaling layout for TMA
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

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

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32, b_ref * b_scale_f32).to(torch.float16)

    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


class TestToMxfp8:
    """Tests for the _to_mxfp8_block library function callable from JIT code with VEC_SIZE=32."""

    @staticmethod
    def _reference_mxfp8_quantize(data, vec_size, torch_dtype):
        """Python reference for MXFP8 quantization matching _compute_scale_and_quantize.

        Note: These tests store the data in SMEM without appropriate prescale swizzling to
        match the assumptions of TMEM. We do not test TMEM directly because we cannot provide
        enough information for an accurate layout.

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
    def _run_to_mxfp8_block(input_data, elem_dtype, device):
        """Run _to_mxfp8_block in a JIT kernel and return FP8 data and scales."""
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
            NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE
            data_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), fp8_type, tl.constexpr(1))
            scale_tile = tlx.local_alloc((BLOCK_M, NUM_SCALES), tl.uint8, tl.constexpr(1))
            data_fp8, scale_e8m0 = tlx._to_mxfp8_block(data, VEC_SIZE, fp8_type)
            tlx.local_store(data_tile[0], data_fp8)
            tlx.local_store(scale_tile[0], scale_e8m0)
            data_fp8 = tlx.local_load(data_tile[0])
            tl.store(data_out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], data_fp8)
            scale_loaded = tlx.local_load(scale_tile[0])
            scale_flat = tl.reshape(scale_loaded, [BLOCK_M * NUM_SCALES])
            tl.store(scale_out_ptr + tl.arange(0, BLOCK_M * NUM_SCALES), scale_flat)

        data_out = torch.empty(M, K, dtype=torch_dtype, device=device)
        scale_out = torch.empty(M * (K // VEC_SIZE), dtype=torch.uint8, device=device)
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
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale.reshape(-1))

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
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale.reshape(-1))
