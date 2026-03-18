import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_hopper
import triton.language.extra.tlx as tlx
from typing import Optional

from test.unit.language.conftest_tlx import _generate_test_params


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
    run_async_dot_blackwell_2cta_tma(device, False, 256)  # A in SMEM
    run_async_dot_blackwell_2cta_tma(device, True, 256)  # A in TMEM

    # M=64 per CTA, explicitly unsupported for now
    # should throw a compilation error for users, but not NE assertion error
    with pytest.raises(Exception) as e:
        run_async_dot_blackwell_2cta_tma(device, False, 128)
    assert isinstance(e.value, triton.CompilationError), "expecting a compilation error"
    assert "only supports M=128 per CTA for pair-CTA mma" in e.value.error_message


def run_async_dot_blackwell_2cta_tma(device, A_TMEM, SAMPLE_M):
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
