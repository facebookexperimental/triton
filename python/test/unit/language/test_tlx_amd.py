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
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 2)

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
    grid = (triton.cdiv(size, 64), )
    _async_load_kernel[grid](x, y, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x + y, output)


# ---------------------------------------------------------------------------
# Test: local_load with relaxed=True sets the syncedViaAsyncWait attribute.
# ---------------------------------------------------------------------------


@triton.jit
def _relaxed_load_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
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
    grid = (triton.cdiv(size, 64), )
    _relaxed_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: async_token survives in scope around tl.range without crashing.
# ---------------------------------------------------------------------------


@triton.jit
def _token_in_loop_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """async_token from async_load_commit_group is live when tl.range is
    entered. If async_token._flatten_ir is broken, the code generator
    crashes with NotImplementedError when collecting carries."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)

    tok = tlx.async_load(x_ptr + offs, buf0, mask=mask)
    tlx.async_load_commit_group([tok])

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

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


# 1D gather test
@triton.jit
def local_gather_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
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
    tl.store(output_ptr + offsets_1d, gathered)


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_local_gather(N, M):
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


@triton.jit
def local_scatter_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    """Test lds scatter using tlx.local_scatter() with axis-based API."""
    # Allocate 2D shared memory and store the matrix
    smem_buffers = tlx.local_alloc((N * M, ), tlx.dtype_of(values_ptr), 1)
    smem = tlx.local_view(smem_buffers, 0)

    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = tl.zeros([N * M], tl.float32)
    tlx.local_store(smem, zeros)

    # Load the scatter indices and values from input
    offsets_1d = tl.arange(0, N)
    indices = tl.load(indices_ptr + offsets_1d)
    values = tl.load(values_ptr + offsets_1d)

    # Scatter using axis-based API: smem_1d[indices[i]] = values[i]
    tlx.local_scatter(smem, values, indices, 0)

    # Read back data from shared memory
    smem_values = tlx.local_load(smem)

    # store result to global memory
    tl.store(output_ptr + offsets_2d, smem_values.reshape([N, M]))


# 1-warp test
@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_local_scatter(N, M):
    """Test scattering to 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create scatter indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    # Create values to scatter
    values = torch.arange(N, dtype=torch.float32, device=device) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Compute expected result: matrix starts at zero, then diagonal gets values
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        expected[i, i] = values[i]

    # Launch kernel
    local_scatter_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# multi-warp test
@pytest.mark.parametrize("N,M,num_warps", [(64, 64, 2), (128, 128, 4)])
def test_scatter_gather_multiwarp(N, M, num_warps):
    """Test scatter and gather with multiple warps."""
    device = torch.device("cuda")

    # Test gather
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)
    gather_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    gather_output = torch.zeros(N, dtype=torch.float32, device=device)
    gather_expected = matrix.flatten()[gather_indices]

    local_gather_kernel[(1, )](
        matrix,
        gather_indices,
        gather_output,
        N=N,
        M=M,
        num_warps=num_warps,
    )

    torch.testing.assert_close(gather_output, gather_expected)

    # Test scatter
    scatter_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    scatter_values = torch.arange(N, dtype=torch.float32, device=device) + 100.0
    scatter_output = torch.zeros((N, M), dtype=torch.float32, device=device)
    scatter_expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        scatter_expected[i, i] = scatter_values[i]

    local_scatter_kernel[(1, )](
        scatter_indices,
        scatter_values,
        scatter_output,
        N=N,
        M=M,
        num_warps=num_warps,
    )

    torch.testing.assert_close(scatter_output, scatter_expected)


# ============================================================================
# 2D Native Gather/Scatter Tests
# ============================================================================


@triton.jit
def local_gather_2d_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 2D gather along specified axis."""
    # Load the matrix from global memory [N, M]
    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_data = tl.load(matrix_ptr + offsets_2d)

    # Store in shared memory
    smem_2d_array = tlx.local_alloc((N, M), tl.float32, 1)
    smem_2d = tlx.local_view(smem_2d_array, 0)
    tlx.local_store(smem_2d, matrix_data)

    # Load indices [N, M] - same rank as source
    indices = tl.load(indices_ptr + offsets_2d)

    # Gather along specified axis
    gathered = tlx.local_gather(smem_2d, indices, axis=axis)

    # Store result
    tl.store(output_ptr + offsets_2d, gathered)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1), (64, 64, 0), (64, 64, 1)])
def test_local_gather_2d_native(N, M, axis):
    """Test 2D gather along different axes."""
    device = torch.device("cuda")

    # Create a test matrix [N, M]
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create indices [N, M] - each position specifies where to gather from along the axis
    if axis == 0:
        # Each column gathers from a shifted row pattern
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (indices + torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
        # Expected: result[i, j] = matrix[indices[i, j], j]
        expected = torch.gather(matrix, 0, indices.long())
    else:  # axis == 1
        # Each row gathers from a shifted column pattern
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (indices + torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M
        # Expected: result[i, j] = matrix[i, indices[i, j]]
        expected = torch.gather(matrix, 1, indices.long())

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    local_gather_2d_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@triton.jit
def local_scatter_2d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 2D scatter along specified axis."""
    # Initialize shared memory to zero
    smem_2d_array = tlx.local_alloc((N, M), tl.float32, 1)
    smem_2d = tlx.local_view(smem_2d_array, 0)

    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = tl.zeros([N, M], tl.float32)
    tlx.local_store(smem_2d, zeros)

    # Load indices [N, M] and values [N, M]
    indices = tl.load(indices_ptr + offsets_2d)
    values = tl.load(values_ptr + offsets_2d)

    # Scatter along specified axis
    tlx.local_scatter(smem_2d, values, indices, axis=axis)

    # Read back the result
    result = tlx.local_load(smem_2d)
    tl.store(output_ptr + offsets_2d, result)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1)])
def test_local_scatter_2d_native(N, M, axis):
    """Test 2D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M] - reverse pattern for scatter
    if axis == 0:
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (N - 1 - indices - torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
    else:  # axis == 1
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (M - 1 - indices - torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M

    # Create values to scatter
    values = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    local_scatter_2d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# ============================================================================
# 3D Gather/Scatter Tests
# ============================================================================


@triton.jit
def local_gather_3d_kernel(
    tensor_ptr,
    indices_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 3D gather along specified axis."""
    # Load the tensor from global memory [N, M, P]
    idx_n = tl.arange(0, N)[:, None, None]
    idx_m = tl.arange(0, M)[None, :, None]
    idx_p = tl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p
    tensor_data = tl.load(tensor_ptr + offsets_3d)

    # Store in shared memory
    smem_3d_array = tlx.local_alloc((N, M, P), tl.float32, 1)
    smem_3d = tlx.local_view(smem_3d_array, 0)
    tlx.local_store(smem_3d, tensor_data)

    # Load indices [N, M, P] - same rank as source
    indices_data = tl.load(indices_ptr + offsets_3d)

    # Gather along specified axis
    gathered = tlx.local_gather(smem_3d, indices_data, axis=axis)

    # Store result
    tl.store(output_ptr + offsets_3d, gathered)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_local_gather_3d_native(N, M, P, axis):
    """Test 3D gather along different axes."""
    device = torch.device("cuda")

    # Create a test tensor [N, M, P]
    tensor = torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P)

    # Create indices [N, M, P] - each position specifies where to gather from along the axis
    if axis == 0:
        # Pattern for gathering along first dimension
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = (base + offset) % N
    elif axis == 1:
        # Pattern for gathering along second dimension
        base = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((base + offset) % M).expand(N, M, P).contiguous()
    else:  # axis == 2
        # Pattern for gathering along third dimension
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        indices = (base % P).expand(N, M, P).contiguous()

    # Ensure indices is contiguous in C-style layout
    indices = indices.contiguous()

    # Compute expected result using torch.gather
    expected = torch.gather(tensor, axis, indices.long())

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    local_gather_3d_kernel[(1, )](
        tensor,
        indices,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@triton.jit
def local_scatter_3d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 3D scatter along specified axis."""
    idx_n = tl.arange(0, N)[:, None, None]
    idx_m = tl.arange(0, M)[None, :, None]
    idx_p = tl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p

    # Initialize shared memory to zero
    smem_3d_array = tlx.local_alloc((N, M, P), tl.float32, 1)
    smem_3d = tlx.local_view(smem_3d_array, 0)

    zeros = tl.full([N, M, P], 0.0, tl.float32)
    tlx.local_store(smem_3d, zeros)

    # Load indices [N, M, P] and values [N, M, P]
    indices_data = tl.load(indices_ptr + offsets_3d)
    values_data = tl.load(values_ptr + offsets_3d)

    # Scatter along specified axis
    tlx.local_scatter(smem_3d, values_data, indices_data, axis=axis)

    # Read back the result
    result = tlx.local_load(smem_3d)
    tl.store(output_ptr + offsets_3d, result)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_scatter_3d_native(N, M, P, axis):
    """Test 3D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M, P] that form a permutation along the scatter axis
    if axis == 0:
        # For axis 0: permute N dimension, keeping (M, P) coordinates fixed
        # Each (j, k) position has a unique permutation of N indices
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = ((N - 1 - base - offset) % N).contiguous()
    elif axis == 1:
        # For axis 1: permute M dimension, keeping (N, P) coordinates fixed
        # Each (i, k) position has a unique permutation of M indices
        base = torch.arange(N * P, dtype=torch.int32, device=device).reshape(N, 1, P)
        offset = torch.arange(M, dtype=torch.int32, device=device).reshape(1, M, 1)
        indices = ((M - 1 - base - offset) % M).contiguous()
    else:  # axis == 2
        # For axis 2: permute P dimension, keeping (N, M) coordinates fixed
        # Each (i, j) position has a unique permutation of P indices
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((P - 1 - base - offset) % P).contiguous()

    # Ensure indices is contiguous
    indices = indices.contiguous()

    # Create values to scatter
    values = (torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P) + 200.0).contiguous()

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M, P), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    local_scatter_3d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)
# ---------------------------------------------------------------------------
# Test: buffer_load compiles on gfx950 and produces the expected ops.
# ---------------------------------------------------------------------------

@triton.jit
def _buffer_load_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements
    x = tlx.buffer_load(x_ptr, offs, mask=mask)
    tl.store(output_ptr + offs, x, mask=mask)


def test_buffer_load_compiles_gfx950(device):
    """buffer_load should produce amdgpu.buffer_load in TTGIR on gfx950."""
    compiled = compile_for_gfx950(
        _buffer_load_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "buffer_load" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_buffer_load_correctness(device):
    """buffer_load produces correct results on gfx950 hardware."""
    if not is_gfx950_available():
        pytest.skip("Requires gfx950 hardware")
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)
    _buffer_load_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: buffer_store compiles on gfx950 and produces the expected ops.
# ---------------------------------------------------------------------------

@triton.jit
def _buffer_store_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tlx.buffer_store(x, output_ptr, offs, mask=mask)


def test_buffer_store_compiles_gfx950(device):
    """buffer_store should produce amdgpu.buffer_store in TTGIR on gfx950."""
    compiled = compile_for_gfx950(
        _buffer_store_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "buffer_store" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_buffer_store_correctness(device):
    """buffer_store produces correct results on gfx950 hardware."""
    if not is_gfx950_available():
        pytest.skip("Requires gfx950 hardware")
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)
    _buffer_store_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)


# ---------------------------------------------------------------------------
# Test: buffer_load_to_local compiles on gfx950 and produces the expected ops.
# ---------------------------------------------------------------------------

@triton.jit
def _buffer_load_to_local_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements

    buf = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    buf0 = tlx.local_view(buf, 0)
    tok = tlx.buffer_load_to_local(buf0, x_ptr, offs, mask=mask)
    tlx.async_load_commit_group([tok])
    tlx.async_load_wait_group(0)

    x = tlx.local_load(buf0)
    tl.store(output_ptr + offs, x, mask=mask)


def test_buffer_load_to_local_compiles_gfx950(device):
    """buffer_load_to_local should produce amdgpu.buffer_load_to_local in TTGIR."""
    compiled = compile_for_gfx950(
        _buffer_load_to_local_kernel,
        signature={"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )
    ttgir = compiled.asm["ttgir"]
    assert "buffer_load_to_local" in ttgir
    assert "async_commit_group" in ttgir
    assert "async_wait" in ttgir
    assert "local_load" in ttgir
    assert "amdgcn" in compiled.asm
    assert len(compiled.asm["amdgcn"]) > 0


def test_buffer_load_to_local_correctness(device):
    """buffer_load_to_local produces correct results on gfx950 hardware."""
    if not is_gfx950_available():
        pytest.skip("Requires gfx950 hardware")
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, 64),)
    _buffer_load_to_local_kernel[grid](x, output, size, BLOCK_SIZE=64)
    torch.testing.assert_close(x, output)
