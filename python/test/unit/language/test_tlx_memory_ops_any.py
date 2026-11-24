"""TLX memory ops tests -- any GPU arch."""
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


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


@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_load_store_smem_with_tl_load(BLOCK_SIZE, device):

    @triton.jit
    def smem_reg_store_load(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        smem_buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 3)
        x_smem = tlx.local_view(smem_buffers, 0)
        y_smem = tlx.local_view(smem_buffers, 1)

        x_tile = tl.load(x_ptr + offsets, mask=mask)
        y_tile = tl.load(y_ptr + offsets, mask=mask)

        tlx.local_store(x_smem, x_tile)
        tlx.local_store(y_smem, y_tile)

        x_reg = tlx.local_load(x_smem)
        y_reg = tlx.local_load(y_smem)
        local_add = x_reg + y_reg
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = smem_reg_store_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 2
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    assert kernel.asm["ttgir"].count("ttg.local_store") == 2
    torch.testing.assert_close(x + y, output)


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_local_gather_1d_native(N, M):
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
