"""Tests that the JIT C dispatcher (_TritonDispatcher) launches through the
shared core triton_launch_kernel() for the converged (non-TMA) path.

The dispatcher is only built when TRITON_USE_C_DISPATCHER=1 (see
CompiledKernel._init_handles). For non-TMA, non multi-dim-cluster kernels the
dispatcher sets use_core_launch and routes td_relaunch through the shared core
(driver.c). These tests force the dispatcher on and verify both end-to-end
numerics and the converged path directly.
"""

import contextlib
import os

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_cuda


@contextlib.contextmanager
def force_dispatcher():
    prev_env = os.environ.get("TRITON_USE_C_DISPATCHER")
    knobs.nvidia.use_triton_dispatcher = True
    os.environ["TRITON_USE_C_DISPATCHER"] = "1"
    try:
        yield
    finally:
        # Clear the cached knob value so subsequent reads fall through to env
        # (assigning prev_knob would leave a stale bool in obj.__dict__).
        del knobs.nvidia.use_triton_dispatcher
        if prev_env is None:
            os.environ.pop("TRITON_USE_C_DISPATCHER", None)
        else:
            os.environ["TRITON_USE_C_DISPATCHER"] = prev_env


@triton.jit
def _disp_add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _disp_scalar_mix(x_ptr, out_ptr, fscale, iadd, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * fscale + iadd, mask=mask)


@triton.jit
def _disp_nop():
    pass


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_core_add_numerics():
    N = 4096
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_dispatcher():
        _disp_add[grid](x, y, out, N, BLOCK=256)
    torch.testing.assert_close(out, x + y)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_core_mixed_scalar_args():
    N = 2048
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
    with force_dispatcher():
        _disp_scalar_mix[grid](x, out, 2.5, 3, N, BLOCK=128)
    torch.testing.assert_close(out, x * 2.5 + 3)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_core_nop():
    with force_dispatcher():
        _disp_nop[(1, )]()
    torch.cuda.synchronize()


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_core_empty_grid():
    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.zeros(N, device="cuda")
    with force_dispatcher():
        _disp_add[(0, )](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros(N, device="cuda"))


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_built_and_converged_path():
    """Directly verify a dispatcher is created (proves the dispatcher path is
    active, not a launchKernel fallback) and that calling it launches correctly
    through the shared core."""
    N = 2048
    BLOCK = 128
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    g = triton.cdiv(N, BLOCK)

    with force_dispatcher():
        compiled = _disp_add.warmup(x, y, out, N, BLOCK=BLOCK, grid=(g, ))
        if hasattr(compiled, "_init_handles"):
            compiled._init_handles()

        disp = getattr(compiled, "_dispatcher", None)
        assert disp is not None, ("expected a C dispatcher when TRITON_USE_C_DISPATCHER=1")

        dev = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(dev)
        # Dispatcher ABI: (grid_x, grid_y, grid_z, stream, *kernel_args)
        disp(g, 1, 1, stream, x, y, out, N)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)


@triton.jit
def _disp_pid_write(out_ptr, GX: tl.constexpr, GY: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    offset = pid_x + GX * (pid_y + GY * pid_z)
    tl.store(out_ptr + offset, offset)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_dispatcher_core_multidim_cluster():
    """An explicit multi-dimensional cluster (ctas_per_cga, which forces
    cluster_dims=(x,y,z) and num_ctas==1) must launch correctly through the
    shared core. With the dispatcher forced on, a non-TMA kernel now sets
    use_core_launch even for multi-dim clusters (previously gated off), so
    triton_launch_kernel builds the explicit
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION. A wrong/missing cluster dim would
    conflict with the PTX .reqnctapercluster baked into the cubin and fail the
    launch; every program writing its linear id proves all CTAs in the
    (2,1,3) cluster ran."""
    from triton._internal_testing import is_hopper_or_newer

    if not is_hopper_or_newer():
        pytest.skip("clusters need Hopper or newer")

    GX, GY, GZ = 4, 2, 3  # each divisible by the (2,1,3) cluster dims
    out = torch.full((GX * GY * GZ, ), -1, device="cuda", dtype=torch.int32)
    with force_dispatcher():
        _disp_pid_write[(GX, GY, GZ)](out, GX, GY, ctas_per_cga=(2, 1, 3))
    torch.cuda.synchronize()
    expected = torch.arange(GX * GY * GZ, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, expected)
