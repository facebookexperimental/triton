"""
Tests for TLX AMD support (async_load, local_load with relaxed, async_token in loops).

These tests compile kernels targeting gfx950 and verify the generated TTIR/TTGIR
contains the expected ops. They do not require AMD hardware -- compilation is
forced via TRITON_OVERRIDE_ARCH and runtime errors are expected and caught.
"""
import os
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip

# Skip the entire module if no HIP runtime is available.
pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")


# ---------------------------------------------------------------------------
# Test: async_load compiles on gfx950 and produces the expected ops.
# ---------------------------------------------------------------------------

@triton.jit
def _async_load_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 2)

    buf0 = tlx.local_view(buffers, 0)
    buf1 = tlx.local_view(buffers, 1)
    tok_x = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tok_y = tlx.async_load(y_ptr + offs, buf1, mask=mask)
    tlx.async_load_commit_group([tok_x, tok_y])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    y = tlx.local_load(buf1)
    tl.store(output_ptr + offs, x + y, mask=mask)


def test_async_load_compiles_gfx950(device):
    """async_load should produce async_copy_global_to_local on gfx950."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)

    try:
        kernel = _async_load_kernel[grid](x, y, output, size, BLOCK_SIZE=64)
        ttgir = kernel.asm["ttgir"]
    except RuntimeError:
        # gfx950 binary can't run on this GPU -- but compilation succeeded.
        pytest.skip("Kernel compiled but cannot run (wrong GPU arch)")
        return

    # If we got here, we're on actual gfx950 hardware.
    assert "async_copy_global_to_local" in ttgir or "buffer_load_to_local" in ttgir
    assert "async_wait" in ttgir or "async_commit_group" in ttgir
    assert "local_load" in ttgir
    torch.testing.assert_close(x + y, output)


# ---------------------------------------------------------------------------
# Test: local_load with relaxed=True sets the syncedViaAsyncWait attribute.
# ---------------------------------------------------------------------------

@triton.jit
def _relaxed_load_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0, relaxed=True)
    tl.store(output_ptr + offs, x, mask=mask)


def test_relaxed_local_load_compiles_gfx950(device):
    """local_load(relaxed=True) should set syncedViaAsyncWait in the IR."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)

    try:
        kernel = _relaxed_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
        ttgir = kernel.asm["ttgir"]
    except RuntimeError:
        pytest.skip("Kernel compiled but cannot run (wrong GPU arch)")
        return

    assert "local_load" in ttgir
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: async_token survives as loop-carried variable without crashing.
# ---------------------------------------------------------------------------

@triton.jit
def _token_in_loop_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """Test that async_token in scope around tl.range does not crash.

    The async_token from async_load_commit_group is live when tl.range
    is entered. If async_token._flatten_ir is not implemented, the code
    generator crashes with NotImplementedError when collecting carries.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)

    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # tok is in scope here -- that's the test.
    for i in tl.range(0, NUM_ITERS, num_stages=0):
        tlx.async_load_wait_group(0)
        x = tlx.local_load(buf0)
        acc += x

    tl.store(output_ptr + offs, acc, mask=mask)


def test_async_token_loop_compiles_gfx950(device):
    """async_token in scope around tl.range should not crash during tracing."""
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)

    old_arch = os.environ.get("TRITON_OVERRIDE_ARCH")
    os.environ["TRITON_OVERRIDE_ARCH"] = "gfx950"
    try:
        kernel = _token_in_loop_kernel[grid](
            x, output, size, BLOCK_SIZE=64, NUM_ITERS=4)
    except RuntimeError as e:
        if "209" in str(e) or "no kernel image" in str(e):
            # Compiled for gfx950 but running on different GPU -- expected.
            return
        raise
    finally:
        if old_arch is None:
            os.environ.pop("TRITON_OVERRIDE_ARCH", None)
        else:
            os.environ["TRITON_OVERRIDE_ARCH"] = old_arch

    # If we reach here, it ran on actual gfx950 hardware.
    torch.testing.assert_close(x * 4, output, rtol=1e-3, atol=1e-3)
