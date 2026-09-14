"""TLX dot tests -- Hopper-only."""
import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton._internal_testing import is_hopper
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _raw_ttir_async_dot_kernel():
    a_tiles = tlx.local_alloc((64, 32), tl.float16, 1)
    b_tiles = tlx.local_alloc((32, 64), tl.float16, 1)
    acc = tlx.async_dot(a_tiles[0], b_tiles[0])
    tlx.async_dot_wait(0, acc)


@triton.jit
def _raw_ttir_ws_async_dot_kernel():
    a_tiles = tlx.local_alloc((64, 32), tl.float16, 1)
    b_tiles = tlx.local_alloc((32, 64), tl.float16, 1)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            _ = tl.arange(0, 1)
        with tlx.async_task(num_warps=4):
            acc = tlx.async_dot(a_tiles[0], b_tiles[0])
            tlx.async_dot_wait(0, acc)


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_raw_ttir_async_dot_has_num_warps():
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 8})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=_raw_ttir_async_dot_kernel, signature={}, constexprs={})

    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )

    assert module.get_int_attr("ttg.num-warps") == 8
    assert module.get_int_attr("ttg.threads-per-warp") == 32
    assert module.get_int_attr("ttg.num-ctas") == 1
    assert module.get_bool_attr("tlx.has_tlx_ops")
    assert module.verify()


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_raw_ttir_async_dot_uses_partition_num_warps():
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 8})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=_raw_ttir_ws_async_dot_kernel, signature={}, constexprs={})

    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    module_text = str(module)

    assert module.get_int_attr("ttg.num-warps") == 8
    assert module_text.count("partition0(") == 1
    assert module_text.count("num_warps(4)") == 1
    assert module_text.count("warpsPerCTA = [4, 1]") == 1
    assert module.verify()


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
def test_async_dot_explicit_accumulator_layout(device):

    @triton.jit
    def _kernel(
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

        a_ptrs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        b_ptrs = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn

        b_alloc = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(Y), 1)
        b_tile = tlx.local_view(b_alloc, 0)

        a_tile = tl.load(a_ptrs)
        tlx.async_load(b_ptrs, b_tile)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        acc_layout: tl.constexpr = tlx.nv_mma_layout(
            warps_per_cta=(2, 2),
            instr_shape=(16, 64, 16),
        )
        acc = tlx.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=acc_layout)
        c = tlx.async_dot(a_tile, b_tile, acc)
        c = tlx.async_dot_wait(tl.constexpr(0), c).to(tlx.dtype_of(Z))
        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.empty((M, N), device=device, dtype=torch.float16)

    kernel = _kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        BLOCK_M=M,
        BLOCK_K=K,
        BLOCK_N=N,
        num_stages=1,
        num_warps=4,
    )

    assert "warpsPerCTA = [2, 2]" in kernel.asm["ttgir"]
    torch.testing.assert_close(z, torch.matmul(x, y))


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_async_dot_explicit_nv_mma_shared_layout(device):

    @triton.jit
    def _kernel(
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

        a_ptrs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        b_ptrs = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn

        b_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
            (BLOCK_K, BLOCK_N // 2),
            [1, 0],
            tlx.dtype_of(Y),
            [1, 1],
            [1, 1],
            [1, 0],
            False,
            True,
        ).make_shared_linear(tile_shape=(BLOCK_K, BLOCK_N))
        b_alloc = tlx.local_alloc(
            (BLOCK_K, BLOCK_N),
            tlx.dtype_of(Y),
            1,
            layout=b_layout,
        )
        b_tile = tlx.local_view(b_alloc, 0)

        a_tile = tl.load(a_ptrs)
        tlx.async_load(b_ptrs, b_tile)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        acc_layout: tl.constexpr = tlx.nv_mma_layout(
            warps_per_cta=(2, 2),
            instr_shape=(16, 64, 16),
        )
        acc = tlx.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=acc_layout)
        c = tlx.async_dot(a_tile, b_tile, acc)
        c = tlx.async_dot_wait(tl.constexpr(0), c).to(tlx.dtype_of(Z))
        c_ptrs = Z + stride_zm * off_m[:, None] + stride_zn * off_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.empty((M, N), device=device, dtype=torch.float16)

    kernel = _kernel[(1, 1)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        z,
        z.stride(0),
        z.stride(1),
        BLOCK_M=M,
        BLOCK_K=K,
        BLOCK_N=N,
        num_stages=1,
        num_warps=4,
    )

    assert "warpsPerCTA = [2, 2]" in kernel.asm["ttgir"]
    assert "#ttg.shared_linear" in kernel.asm["ttgir"]
    torch.testing.assert_close(z, torch.matmul(x, y))


@pytest.mark.skipif(not is_hopper(), reason="Need Hopper")
def test_memdesc_trans_sliced_shared_linear_layout(device):

    @triton.jit
    def _kernel(X, K, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
        NUM_MMA_GROUPS: tl.constexpr = 2
        CID_BLOCK_N: tl.constexpr = BLOCK_N // NUM_MMA_GROUPS

        off_n = tl.arange(0, CID_BLOCK_N)
        off_m = tl.arange(0, BLOCK_M)
        off_d = tl.arange(0, HEAD_DIM)

        score_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
            (CID_BLOCK_N, BLOCK_M),
            [1, 0],
            tlx.dtype_of(X),
            [1, 1],
            [1, 1],
            [1, 0],
            False,
            True,
        ).tile_to_shape((BLOCK_N, BLOCK_M))
        k_layout: tl.constexpr = tlx.nv_mma_shared_layout_encoding(
            (CID_BLOCK_N, HEAD_DIM // NUM_MMA_GROUPS),
            [1, 0],
            tlx.dtype_of(K),
            [1, 1],
            [1, 1],
            [1, 0],
            False,
            True,
        ).tile_to_shape((BLOCK_N, HEAD_DIM))

        score_full = tlx.local_alloc((BLOCK_N, BLOCK_M), tlx.dtype_of(X), 1, layout=score_layout)
        k_full = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), 1, layout=k_layout)
        score = tlx.local_slice(score_full[0], [0, 0], [CID_BLOCK_N, BLOCK_M])
        score_t = tlx.local_trans(score)
        k_tile = tlx.local_slice(k_full[0], [0, 0], [CID_BLOCK_N, HEAD_DIM])

        x = tl.load(X + off_n[:, None] * BLOCK_M + off_m[None, :])
        k = tl.load(K + off_n[:, None] * HEAD_DIM + off_d[None, :])
        tlx.local_store(score, x)
        tlx.local_store(k_tile, k)
        tlx.fence_async_shared()

        z = tlx.async_dot(score_t, k_tile)
        z = tlx.async_dot_wait(0, z)
        tl.store(Z + off_m[:, None] * HEAD_DIM + off_d[None, :], z)

    torch.manual_seed(0)
    block_m = 64
    block_n = 128
    head_dim = 128
    cid_block_n = block_n // 2
    x = torch.randn((cid_block_n, block_m), device=device, dtype=torch.bfloat16)
    k = torch.randn((cid_block_n, head_dim), device=device, dtype=torch.bfloat16)
    z = torch.empty((block_m, head_dim), device=device, dtype=torch.float32)

    kernel = _kernel[(1, 1)](
        x,
        k,
        z,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_stages=1,
        num_warps=4,
    )

    ttgir = kernel.asm["ttgir"]
    assert "ttg.memdesc_trans" in ttgir
    assert "ttg.local_load" not in ttgir
    torch.testing.assert_close(z, x.T.float() @ k.float(), atol=0.2, rtol=0.2)


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

        a_view = tlx.local_view(a_tiles, 0)
        b_view = tlx.local_view(b_tiles, 0)

        a_full = tlx.local_view(a_fulls, 0)
        tlx.barrier_expect_bytes(a_full, 2 * BLOCK * BLOCK)
        tlx.async_descriptor_load(desc_a, a_view, [0, 0], a_full)
        b_full = tlx.local_view(b_fulls, 0)
        tlx.barrier_expect_bytes(b_full, 2 * BLOCK * BLOCK)
        tlx.async_descriptor_load(desc_b, b_view, [0, 0], b_full)

        tlx.barrier_wait(a_full, 0)
        tlx.barrier_wait(b_full, 0)
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

    _kernel[(1, )](desc_a, desc_b, desc_c, BLOCK=BLOCK, num_stages=1, num_warps=4)
    z_ref = torch.matmul(a, b)
    torch.testing.assert_close(c, z_ref)
