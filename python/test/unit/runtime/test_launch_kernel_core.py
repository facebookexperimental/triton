"""Tests that the JIT variadic launcher (driver.c::launchKernel) routes through
the shared data-driven core ``triton_launch_kernel`` in ``triton/runtime/launch.h``.

The default launch path (no ``TRITON_USE_C_DISPATCHER``) goes through
``kernel.run() -> CudaLauncher -> launchKernel``. To make these tests robust
regardless of defaults, each test installs a ``launch_enter_hook``: jit.py only
takes the C-dispatcher / c_cache fast paths when *no* launch hooks are set, so a
hook guarantees the ``launchKernel`` path and — because the hook is invoked
*inside* ``launchKernel`` — a non-zero hook-call count proves that path ran.
"""

import contextlib

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_cuda


@contextlib.contextmanager
def force_launch_kernel_path():
    """Install a launch_enter_hook to force (and prove) the launchKernel path."""
    counter = {"calls": 0}

    def _enter(_metadata):
        counter["calls"] += 1

    prev = knobs.runtime.launch_enter_hook
    knobs.runtime.launch_enter_hook = _enter
    try:
        yield counter
    finally:
        knobs.runtime.launch_enter_hook = prev


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _nop_kernel():
    pass


@triton.jit
def _scalar_mix_kernel(x_ptr, out_ptr, fscale, iadd, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * fscale + iadd, mask=mask)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_launchkernel_path_add_numerics():
    N = 4096
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_launch_kernel_path() as counter:
        _add_kernel[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y)
    assert counter["calls"] > 0, "launchKernel path was not exercised"


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_launchkernel_path_nop_zero_args():
    with force_launch_kernel_path() as counter:
        _nop_kernel[(1, )]()
    torch.cuda.synchronize()
    assert counter["calls"] > 0, "launchKernel path was not exercised"


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_launchkernel_path_mixed_scalar_args():
    N = 2048
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_launch_kernel_path() as counter:
        _scalar_mix_kernel[grid](x, out, 2.5, 3, N, BLOCK=128)
    torch.testing.assert_close(out, x * 2.5 + 3)
    assert counter["calls"] > 0, "launchKernel path was not exercised"


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_launchkernel_path_empty_grid():
    """An empty (0) grid must be a no-op, not a CUDA error — the shared core
    skips a zero-sized grid (parity with the legacy JIT _launch)."""
    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.zeros(N, device="cuda")
    with force_launch_kernel_path() as counter:
        _add_kernel[(0, )](x, y, out, N, BLOCK=256)  # empty grid: no work
    torch.cuda.synchronize()
    # out must be untouched (kernel never ran)
    torch.testing.assert_close(out, torch.zeros(N, device="cuda"))
    assert counter["calls"] > 0, "launchKernel path was not exercised"


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_launchkernel_path_host_tma_passthrough():
    """Host-side TMA: Python builds the CUtensorMap; launchKernel passes it
    through args_buf as an ordinary (is_tma=0) param to the shared core."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    M, N = 128, 128
    M_BLOCK, N_BLOCK = 64, 64

    @triton.jit
    def _td_kernel(out_ptr, desc, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        block = desc.load([0, 0])
        idx = (tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :])
        tl.store(out_ptr + idx, block)

    t = torch.randn((M, N), device="cuda")
    out = torch.empty((M_BLOCK, N_BLOCK), device="cuda")
    desc = TensorDescriptor(t, shape=t.shape, strides=t.stride(), block_shape=[M_BLOCK, N_BLOCK])
    with force_launch_kernel_path() as counter:
        _td_kernel[(1, )](out, desc, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(out, t[:M_BLOCK, :N_BLOCK])
    assert counter["calls"] > 0, "launchKernel path was not exercised"
