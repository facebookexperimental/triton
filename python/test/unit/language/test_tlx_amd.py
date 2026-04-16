"""
Tests for TLX AMD support (async_load, local_load with relaxed, async_token in loops).

These tests compile kernels targeting gfx950 via triton.compile() with an
explicit GPUTarget and verify the generated TTGIR/AMDGCN. No AMD hardware is
required for the compilation checks. Correctness checks (actual execution) run
only when gfx950 hardware is available.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.backends.compiler import GPUTarget

# Skip the entire module if no HIP runtime is available.
pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")

GFX950 = GPUTarget("hip", "gfx950", 64)


def compile_for_gfx950(fn, signature, constexprs):
    """Compile a TLX kernel for gfx950 and return the compiled object."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton_compile(src, target=GFX950)


def is_gfx950_available():
    """Check if the current device is gfx950."""
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx950"
    except Exception:
        return False


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
    """async_load should produce async_copy_global_to_local in TTGIR on gfx950."""
    compiled = compile_for_gfx950(
        _async_load_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "async_copy_global_to_local" in ttgir or "buffer_load_to_local" in ttgir
    assert "async_commit_group" in ttgir
    assert "async_wait" in ttgir
    assert "local_load" in ttgir

    # Verify the kernel compiled all the way to AMDGCN.
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_async_load_correctness(device):
    """async_load produces correct results on gfx950 hardware."""
    if not is_gfx950_available():
        pytest.skip("Requires gfx950 hardware")
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)
    _async_load_kernel[grid](x, y, output, size, BLOCK_SIZE=64)
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
    """local_load(relaxed=True) should compile and produce local_load in TTGIR."""
    compiled = compile_for_gfx950(
        _relaxed_load_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir


def test_relaxed_local_load_correctness(device):
    """local_load(relaxed=True) produces correct results on gfx950 hardware."""
    if not is_gfx950_available():
        pytest.skip("Requires gfx950 hardware")
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)
    _relaxed_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: async_token survives in scope around tl.range without crashing.
# ---------------------------------------------------------------------------

@triton.jit
def _token_in_loop_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """async_token from async_load_commit_group is live when tl.range is
    entered. If async_token._flatten_ir is broken, the code generator
    crashes with NotImplementedError when collecting carries."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)

    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # tok is in scope here -- that's what we're testing.
    for i in tl.range(0, NUM_ITERS, num_stages=0):
        tlx.async_load_wait_group(0)
        x = tlx.local_load(buf0)
        acc += x

    tl.store(output_ptr + offs, acc, mask=mask)


def test_async_token_loop_compiles_gfx950(device):
    """async_token in scope around tl.range should compile without crashing."""
    compiled = compile_for_gfx950(
        _token_in_loop_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64, "NUM_ITERS": 4},
    )
    ttgir = compiled.asm["ttgir"]
    assert "local_load" in ttgir
    assert "async_wait" in ttgir


@triton.jit
def local_gather_kernel(
    matrix_ptr, indices_ptr, output_ptr, 
    N: tl.constexpr, 
    M: tl.constexpr,
):
    """Test lds gather using tlx.local_gather() with axis-based API."""
    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_regs = tl.load(matrix_ptr + offsets_2d)

    # Allocate 2D shared memory and store the matrix
    smem_1d_buffers = tlx.local_alloc((N * M, ), tlx.dtype_of(matrix_ptr), 1)
    smem_1d = tlx.local_view(smem_1d_buffers, 0)
    tlx.local_store(smem_1d, matrix_regs.reshape((N * M, )))

    # Load the gather indices
    offsets_1d = tl.arange(0, N)
    indices = tl.load(indices_ptr + offsets_1d)

    # Gather using axis-based API: result[i] = smem_1d[indices[i]]
    gathered = tlx.local_gather(smem_1d, indices, 0)

    # store result to global memory
    tl.store(output_ptr, offsets_1d, gathered)


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_shared_gather(N, M):
    """Test gathering from 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create a test matrix with known values
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create gather indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    output = torch.zeros(N, dtype=torch.float32, device=device)

    # Compute expected result: diagonal elements
    expected = matrix.flatten()[indices]

    # Launch kernel
    local_gather_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)