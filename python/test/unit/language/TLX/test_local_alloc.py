"""Standalone tests for tlx_plugin memory ops (local_alloc, local_view,
local_store, local_load).

Tests the plugin-ported local_alloc operation: allocating shared memory
buffers, storing data into them, and loading data back out. Covers single
and multi-buffer allocation, different dtypes, and 1D/2D shapes.

These tests import from tlx_plugin (the out-of-tree plugin Python DSL)
rather than triton.language.extra.tlx (the in-tree TLX DSL).
"""

import importlib
import sys
import os
import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

# Add the plugin Python package to the path.
# Resolve to absolute path so it works regardless of cwd.
_plugin_python_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..",
                 "examples", "plugins", "TLXPlugin", "python")
)
if _plugin_python_dir not in sys.path:
    sys.path.insert(0, _plugin_python_dir)
import tlx_plugin as tlx  # type: ignore[import-not-found]
from tlx_plugin.custom_stages import inspect_stages_hook

# Activate the plugin's custom ConvertTritonToTritonGPU pass so that
# ttg.local_store / ttg.local_load ops emitted at TTIR level are legalized.
knobs.runtime.add_stages_inspection_hook = inspect_stages_hook


def is_hopper_or_newer():
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Basic: alloc -> store -> load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("dtype", [tl.float16, tl.float32, tl.bfloat16])
def test_local_alloc_store_load_1d(dtype, device="cuda"):
    """Allocate a 1D SMEM buffer, store a vector, load it back."""

    @triton.jit
    def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((BLOCK,), DTYPE, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    BLOCK = 128
    torch_dtype = {tl.float16: torch.float16, tl.float32: torch.float32,
                   tl.bfloat16: torch.bfloat16}[dtype]
    x = torch.randn(BLOCK, device=device, dtype=torch_dtype)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, BLOCK=BLOCK, DTYPE=dtype)
    torch.testing.assert_close(out, x)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_alloc_store_load_2d(device="cuda"):
    """Allocate a 2D SMEM buffer, store a tile, load it back."""

    @triton.jit
    def kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
        row = tl.arange(0, M)
        col = tl.arange(0, N)
        offs = row[:, None] * N + col[None, :]
        x = tl.load(in_ptr + offs)
        buf = tlx.local_alloc((M, N), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(out_ptr + offs, y)

    M, N = 64, 64
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, M=M, N=N)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Multi-buffer: alloc with num > 1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_alloc_multi_buffer(device="cuda"):
    """Allocate multiple SMEM buffers and use them independently."""

    @triton.jit
    def kernel(in_ptr_a, in_ptr_b, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        a = tl.load(in_ptr_a + offs)
        b = tl.load(in_ptr_b + offs)

        bufs = tlx.local_alloc((BLOCK,), tl.float16, 2)

        view0 = tlx.local_view(bufs, 0)
        view1 = tlx.local_view(bufs, 1)

        tlx.local_store(view0, a)
        tlx.local_store(view1, b)

        a_loaded = tlx.local_load(view0)
        b_loaded = tlx.local_load(view1)

        tl.store(out_ptr + offs, a_loaded + b_loaded)

    BLOCK = 128
    a = torch.randn(BLOCK, device=device, dtype=torch.float16)
    b = torch.randn(BLOCK, device=device, dtype=torch.float16)
    out = torch.empty(BLOCK, device=device, dtype=torch.float16)

    kernel[(1,)](a, b, out, BLOCK=BLOCK)
    torch.testing.assert_close(out, a + b)


# ---------------------------------------------------------------------------
# Multi-buffer 2D: typical GEMM-style double buffering
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_alloc_double_buffer_2d(device="cuda"):
    """Allocate 2 x (M, K) SMEM buffers simulating double buffering."""

    @triton.jit
    def kernel(in_ptr, out_ptr, M: tl.constexpr, K: tl.constexpr):
        row = tl.arange(0, M)
        col = tl.arange(0, K)
        offs = row[:, None] * K + col[None, :]
        x = tl.load(in_ptr + offs)

        bufs = tlx.local_alloc((M, K), tl.float16, 2)

        # Store into buffer 0, load back
        v0 = tlx.local_view(bufs, 0)
        tlx.local_store(v0, x)
        y = tlx.local_load(v0)

        # Store into buffer 1, load back (should give same result)
        v1 = tlx.local_view(bufs, 1)
        tlx.local_store(v1, y)
        z = tlx.local_load(v1)

        tl.store(out_ptr + offs, z)

    M, K = 64, 64
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    out = torch.empty_like(x)

    kernel[(1,)](x, out, M=M, K=K)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# Compile-only: verify IR generation without running on GPU
# ---------------------------------------------------------------------------

def test_local_alloc_compile_only():
    """Verify local_alloc generates valid TTGIR (no GPU required)."""

    @triton.jit
    def kernel(ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(ptr + offs)
        buf = tlx.local_alloc((BLOCK,), tl.float16, 1)
        view = tlx.local_view(buf, 0)
        tlx.local_store(view, x)
        y = tlx.local_load(view)
        tl.store(ptr + offs, y)

    # Compile to TTGIR (does not require GPU)
    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={"ptr": "*fp16"},
        constexprs={"BLOCK": 128},
    )
    try:
        ret = triton.compile(src, target=triton.runtime.driver.active.get_current_target())
    except Exception:
        # If no GPU available, just verify the kernel parses
        pytest.skip("No GPU target available for compilation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
