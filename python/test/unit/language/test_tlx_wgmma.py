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
    """
    Define a unit test with similar schema with tl.dot
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
    """

    @triton.jit
    def tgt_kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr, out_dtype: tl.constexpr,
                   COL_INPUT: tl.constexpr, COL_OTHER: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)

        buf_alloc_x = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float32, 1)
        buf_alloc_y = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float32, 1)
        a_smem = tlx.local_view(buf_alloc_x, 0)
        b_smem = tlx.local_view(buf_alloc_y, 0)

        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn

        # TODO. initialize values or async load

        z = tlx.async_dot(a_smem, b_smem, input_precision=INPUT_PRECISION, out_dtype=out_dtype, col_input=COL_INPUT,
                          col_other=COL_OTHER)
        tl.store(Zs, z)

    M, N, K = (64, 64, 64)
    x = torch.ones((M, K), device=device, dtype=torch.float32)
    y = torch.ones((K, N), device=device, dtype=torch.float32)
    z = torch.empty_like(x, device=device, dtype=torch.float32)

    kern_kwargs = {
        'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'INPUT_PRECISION': "tf32", 'out_dtype': tl.float32, 'COL_INPUT': 0,
        'COL_OTHER': 1
    }
    with pytest.raises(RuntimeError) as _:
        _ = tgt_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0), z.stride(1),
                               **kern_kwargs)

    # TODO. assert "ttng.warp_group_dot" in pgm["ttir"] but not accessible due to thrown RuntimeError
    # Following snippet can be found in the printed TTIR.
    # %49 = "tlx.require_layout"(%10) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    # %50 = "tlx.require_layout"(%13) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>
    # %51 = "ttg.convert_layout"(%3) : (tensor<64x64xf32>) -> tensor<64x64xf32, #mma>
    # "ttng.fence_async_shared"() <{bCluster = false}> : () -> ()
    # %52 = "ttng.warp_group_dot"(%49, %50, %51) <{inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32}> : (!ttg.memdesc<64x64xf32, #shared1, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared2, #smem, mutable>, tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma>
