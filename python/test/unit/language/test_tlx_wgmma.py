
import pytest
import torch
import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton._internal_testing import is_cuda

@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
def test_async_dot(device):
    # Define a unit test with similar schema with tl.dot

    @triton.jit
    def ref_kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr, out_dtype: tl.constexpr = tl.float32):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)

    @triton.jit
    def tgt_kernel(
        X, stride_xm, stride_xk, 
        Y, stride_yk, stride_yn, 
        Z, stride_zm, stride_zn,
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
        BLOCK_K: tl.constexpr, 
        INPUT_PRECISION: tl.constexpr, 
        out_dtype: tl.constexpr = tl.float32
        # a_ptr,
        # b_ptr,
        # stride_cm,
        # stride_cn,
        # BLOCK_SIZE: tl.constexpr,
    ):
        dummy_output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)

        buf_alloc_x = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float32, 1)
        buf_alloc_y = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float32, 1)
        a_smem = tlx.local_view(buf_alloc_x, 0)
        b_smem = tlx.local_view(buf_alloc_y, 0)

        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn

        # TODO. initialize values or async load

        z = tlx.async_dot(a_smem, b_smem, dummy_output, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)

        # offs_cm = tl.arange(0, BLOCK_SIZE)
        # offs_cn = tl.arange(0, BLOCK_SIZE)
        # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # tl.store(c_ptrs, acc)

    # a_gmem = torch.ones((64, 64), device=device, dtype=torch.float32)
    # b_gmem = torch.ones((64, 64), device=device, dtype=torch.float32)
    # c = torch.ones((64, 64), device=device, dtype=torch.float32)
    # grid = lambda META: (1,)
    # async_dot_kernel[grid](
    #     # a_gmem,
    #     # b_gmem,
    #     c,  #
    #     c.stride(0),
    #     c.stride(1),  #
    #     64,
    #     num_warps=4,
    # )

    # print("\nc = ", c)
    # assert False

    M,N,K = (64,64,64)
    x = torch.ones((M, K), device=device, dtype=torch.float32)
    y = torch.ones((K, N), device=device, dtype=torch.float32)
    z = torch.empty_like(x, device=device, dtype=torch.float32)

    kern_kwargs = {
        'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'INPUT_PRECISION': "tf32", 'out_dtype': tl.float32
    }
    pgm = tgt_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0), z.stride(1), **kern_kwargs)

    assert False
