import itertools
import pytest
import torch
import triton.runtime.driver as driver


def _generate_test_params():
    """Generate test parameters with filtering for memory constraints."""
    dims_mn = [16, 32, 64, 128, 512]
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
            # This shape incurs excessive register pressure and fails on H100
            params.append(pytest.param(M, N, K, marks=pytest.mark.xfail()))
        else:
            params.append((M, N, K))
    return params


def _swizzle_scale_to_5d(scale, outer_chunks, k_chunks):
    """Convert raw E8M0 scales to swizzled 5D format for TMA/async_dot_scaled.

    Applies the cuBLAS block scaling layout within each 128x4 block.
    dest[row%32 * 16 + row//32 * 4 + col] = src[row, col]

    Args:
        scale: Raw scale tensor of shape (batch, rows, K//32) in uint8.
        outer_chunks: Number of 128-row chunks (rows // 128).
        k_chunks: Number of 4-column chunks (K // 32 // 4).

    Returns:
        Swizzled 5D tensor of shape (batch, outer_chunks, k_chunks, 2, 256).
    """
    batch = scale.shape[0]
    cols = scale.shape[2]
    padded_cols = k_chunks * 4

    if cols < padded_cols:
        scale = torch.nn.functional.pad(scale, (0, padded_cols - cols))

    blocks = (scale.reshape(batch, outer_chunks, 128, k_chunks,
                            4).permute(0, 1, 3, 2, 4).reshape(batch, outer_chunks, k_chunks, 512))

    _r = torch.arange(128)
    _c = torch.arange(4)
    _rg, _cg = torch.meshgrid(_r, _c, indexing="ij")
    idx = ((_rg % 32) * 16 + (_rg // 32) * 4 + _cg).reshape(-1)
    idx = idx.to(scale.device).expand_as(blocks)
    output = torch.empty_like(blocks)
    output.scatter_(-1, idx, blocks)

    return output.reshape(batch, outer_chunks, k_chunks, 2, 256)
