"""Tests for the ctypes-based no-compile launcher.

Verifies that kernels launched via the ctypes launcher (TRITON_USE_NO_COMPILE_LAUNCHER=1)
produce identical results to the default C-compiled launcher. Tests cover:
1. Regular kernels (no tensor descriptors)
2. Host-side tensor descriptors (tensordesc_meta entries are None)
3. Device-side TMA tensor descriptors (tensordesc_meta entries are dicts)
"""

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import (
    is_cuda,
    requires_tma,
)
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


# ---------------------------------------------------------------------------
# 2. Host-side tensor descriptor
# ---------------------------------------------------------------------------


@triton.jit(debug=True)
def _host_tensordesc_load_kernel(
    out_ptr, desc, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr
):
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

    inp_desc = TensorDescriptor(
        inp, shape=inp.shape, strides=inp.stride(), block_shape=[M_BLOCK, N_BLOCK]
    )

    # Run with C launcher
    out_c = torch.empty((M_BLOCK, N_BLOCK), device=device, dtype=torch.float16)
    _host_tensordesc_load_kernel[(1,)](out_c, inp_desc, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(out_c, expected)

    # Clear cache and run with ctypes launcher
    _host_tensordesc_load_kernel.device_caches.clear()

    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        out_ctypes = torch.empty(
            (M_BLOCK, N_BLOCK), device=device, dtype=torch.float16
        )
        _host_tensordesc_load_kernel[(1,)](
            out_ctypes, inp_desc, M, N, M_BLOCK, N_BLOCK
        )

    torch.testing.assert_close(out_ctypes, expected)
    torch.testing.assert_close(out_ctypes, out_c)


# ---------------------------------------------------------------------------
# 3. Device-side TMA tensor descriptor
# ---------------------------------------------------------------------------


@triton.jit
def _tma_tensordesc_load_kernel(
    out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr
):
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
    _tma_tensordesc_load_kernel[(1,)](out_c, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(out_c, expected)

    # Clear cache and run with ctypes launcher
    _tma_tensordesc_load_kernel.device_caches.clear()

    with knobs.nvidia.scope():
        knobs.nvidia.use_no_compile_launcher = True
        out_ctypes = torch.empty(
            (M_BLOCK, N_BLOCK), device=device, dtype=torch.float16
        )
        _tma_tensordesc_load_kernel[(1,)](out_ctypes, inp, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(out_ctypes, expected)
    torch.testing.assert_close(out_ctypes, out_c)
