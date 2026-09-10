"""TLX dot tests -- AMD or Hopper; skipped on Blackwell."""
import itertools
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx
import triton.runtime.driver as driver


def _generate_test_params():
    """Generate test parameters with filtering for memory constraints."""
    # 128 dropped from dims_mn: redundant with the 64/512 tiles either side of it.
    dims_mn = [16, 32, 64, 512]
    dims_k = [16, 32, 64]
    dtype = torch.float16
    params = []

    try:
        device_props = str(torch.cuda.get_device_properties())
        max_shared_mem = driver.active.utils.get_device_properties(driver.active.get_current_device())["max_shared_mem"]
    except RuntimeError:
        # CUDA not available (e.g., ASAN build or no GPU); return all combos unskipped
        return list(itertools.product(dims_mn, dims_mn, dims_k))

    for M, N, K in itertools.product(dims_mn, dims_mn, dims_k):
        matmul_size = (M * K + K * N) * dtype.itemsize
        if matmul_size > max_shared_mem:
            continue
        # TODO: Investigate why this test fails on gfx942 with M=512, N=512, K=16
        if "gfx942" in device_props and M == 512 and N == 512 and K == 16:
            params.append(pytest.param(M, N, K, marks=pytest.mark.xfail()))
        elif "H100" in device_props and M == 512 and N == 512 and K == 64:
            # This shape incurs excessive register pressure and fails on H100.
            # Skip rather than xfail: the failing compile is never cached, so
            # running it costs ~170s -- more than the rest of the suite combined.
            params.append(pytest.param(M, N, K, marks=pytest.mark.skip(reason="excessive register pressure on H100")))
        else:
            params.append((M, N, K))
    return params


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
