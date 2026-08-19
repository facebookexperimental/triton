"""Tests for the ctypes-based no-compile launcher.

Verifies that kernels launched via the ctypes launcher (TRITON_USE_NO_COMPILE_LAUNCHER=1)
produce identical results to the default C-compiled launcher. Tests cover:
1. Regular kernels (no tensor descriptors)
2. Host-side tensor descriptors (tensordesc_meta entries are None)
3. Device-side TMA tensor descriptors (tensordesc_meta entries are dicts)
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import (
    is_cuda,
    is_hopper_or_newer,
    requires_tma,
)
from triton.backends.nvidia.ctypes_launcher import _validate_exact_cluster_grid
from triton.tools.tensor_descriptor import TensorDescriptor


def _skip_if_not_cuda():
    if not is_cuda():
        pytest.skip("ctypes launcher requires CUDA")


# ---------------------------------------------------------------------------
# 1. Regular kernel (no tensor descriptors)
# ---------------------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def test_no_compile_launcher_add(device, fresh_triton_cache):
    _skip_if_not_cuda()

    N = 1024
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.randn(N, device=device, dtype=torch.float32)
    expected = x + y

    # Run with C launcher (default)
    out_c = torch.empty_like(x)
    _add_kernel[(N // 256, )](x, y, out_c, N, BLOCK=256)
    torch.testing.assert_close(out_c, expected)

    # Clear cache to force re-compilation with ctypes launcher
    _add_kernel.device_caches.clear()

    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        out_ctypes = torch.empty_like(x)
        _add_kernel[(N // 256, )](x, y, out_ctypes, N, BLOCK=256)

    torch.testing.assert_close(out_ctypes, expected)
    torch.testing.assert_close(out_ctypes, out_c)


@triton.jit
def _pid_write(out_ptr, GX: tl.constexpr, GY: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    offset = pid_x + GX * (pid_y + GY * pid_z)
    tl.store(out_ptr + offset, offset)


@pytest.mark.parametrize(
    "num_ctas,cluster_dims",
    [
        (0, (1, 1, 1)),
        (1, (2, 0, 2)),
        (1, (2, -1, 2)),
        (1, (2, 2)),
    ],
)
def test_no_compile_launcher_rejects_invalid_cluster_metadata(num_ctas, cluster_dims):
    with pytest.raises(ValueError, match="Invalid Triton cluster metadata"):
        _validate_exact_cluster_grid((4, 2, 4), num_ctas, cluster_dims)


def test_no_compile_launcher_multidim_cluster(device, fresh_triton_cache):
    _skip_if_not_cuda()
    if not is_hopper_or_newer():
        pytest.skip("clusters need Hopper or newer")

    grid = (4, 2, 4)
    out = torch.full((grid[0] * grid[1] * grid[2], ), -1, device=device, dtype=torch.int32)
    _pid_write.device_caches.clear()
    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        _pid_write[grid](out, grid[0], grid[1], ctas_per_cga=(2, 2, 2))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.arange(out.numel(), device=device, dtype=torch.int32))


@pytest.mark.parametrize("grid", [(3, 2, 4), (4, 3, 4), (4, 2, 3)])
def test_no_compile_launcher_rejects_incomplete_cluster(device, fresh_triton_cache, grid):
    _skip_if_not_cuda()
    if not is_hopper_or_newer():
        pytest.skip("clusters need Hopper or newer")

    out = torch.empty((grid[0] * grid[1] * grid[2], ), device=device, dtype=torch.int32)
    _pid_write.device_caches.clear()
    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        with pytest.raises(
            ValueError,
            match=rf"physical grid {re.escape(str(grid))}.*required cluster shape \(2, 2, 2\)",
        ):
            _pid_write[grid](out, grid[0], grid[1], ctas_per_cga=(2, 2, 2))


# ---------------------------------------------------------------------------
# 2. Host-side tensor descriptor
# ---------------------------------------------------------------------------


@triton.jit(debug=True)
def _host_tensordesc_load_kernel(out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


@requires_tma
def test_no_compile_launcher_host_tensordesc(device, fresh_triton_cache):
    _skip_if_not_cuda()

    M_BLOCK, N_BLOCK = 8, 32
    M, N = M_BLOCK * 3, N_BLOCK * 4
    inp = torch.randn((M, N), device=device, dtype=torch.float16)
    expected = inp[:M_BLOCK, :N_BLOCK].clone()

    inp_desc = TensorDescriptor(inp, shape=inp.shape, strides=inp.stride(), block_shape=[M_BLOCK, N_BLOCK])

    # Run with C launcher
    out_c = torch.empty((M_BLOCK, N_BLOCK), device=device, dtype=torch.float16)
    _host_tensordesc_load_kernel[(1, )](out_c, inp_desc, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(out_c, expected)

    # Clear cache and run with ctypes launcher
    _host_tensordesc_load_kernel.device_caches.clear()

    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        out_ctypes = torch.empty((M_BLOCK, N_BLOCK), device=device, dtype=torch.float16)
        _host_tensordesc_load_kernel[(1, )](out_ctypes, inp_desc, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(out_ctypes, expected)
    torch.testing.assert_close(out_ctypes, out_c)


# ---------------------------------------------------------------------------
# 3. Device-side TMA tensor descriptor
# ---------------------------------------------------------------------------


@triton.jit
def _tma_tensordesc_load_kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    block = desc.load([0, 0])
    idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
    tl.store(out_ptr + idx, block)


@requires_tma
def test_no_compile_launcher_tma_tensordesc(device, fresh_triton_cache, with_allocator):
    _skip_if_not_cuda()

    M_BLOCK, N_BLOCK = 8, 32
    M, N = M_BLOCK * 3, N_BLOCK * 4
    inp = torch.randn((M, N), device=device, dtype=torch.float16)
    expected = inp[:M_BLOCK, :N_BLOCK].clone()

    # Run with C launcher
    out_c = torch.empty((M_BLOCK, N_BLOCK), device=device, dtype=torch.float16)
    _tma_tensordesc_load_kernel[(1, )](out_c, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(out_c, expected)

    # Clear cache and run with ctypes launcher
    _tma_tensordesc_load_kernel.device_caches.clear()

    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        out_ctypes = torch.empty((M_BLOCK, N_BLOCK), device=device, dtype=torch.float16)
        _tma_tensordesc_load_kernel[(1, )](out_ctypes, inp, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(out_ctypes, expected)
    torch.testing.assert_close(out_ctypes, out_c)
