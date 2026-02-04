"""
Triton implementation of to_mxfp8 conversion for use inside Triton kernels.

This module provides a Triton JIT function to convert float32 tensors to MXFP8 format
(FP8 E4M3 data + E8M0 scales), with the output scale swizzled for TMEM usage.

The implementation follows the RCEIL rounding mode from the torchao MXFP8 implementation.
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# PyTorch host-side MXFP8 conversion functions
# (Extracted from blackwell-fa-ws-pipelined-persistent_mxfp8_test.py)
# =============================================================================


# This function is extracted from https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142
def to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 32,
):
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert data_hp.shape[-1] % block_size == 0, (
        f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}")
    assert data_hp.is_contiguous(), "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    max_pos = F8E4M3_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
            (torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            ) + E8M0_EXPONENT_BIAS).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    data_lp = data_lp.to(torch.float8_e4m3fn)
    # need to reshape at the end to help inductor fuse things
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp


# =============================================================================
# Shared swizzle function for 5D scale layout
# =============================================================================


@triton.jit
def swizzle_scales_block(
    scale_input,  # [BLOCK_ROWS, BLOCK_COLS] tensor (already loaded)
    scale_output,  # Output pointer
    output_offset,  # Offset in output buffer for this block
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """
    Swizzle a block of scales to 5D format and store to output.

    The swizzling follows NVIDIA's block scaling factors layout:
    - 128 rows are grouped into 4 sub-blocks of 32 rows
    - Swizzle formula: dest_idx = (r % 32) * 16 + (r // 32) * 4 + c

    Args:
        scale_input: Loaded scale tensor [BLOCK_ROWS, BLOCK_COLS]
        scale_output: Output buffer pointer
        output_offset: Offset in output buffer for this block
        BLOCK_ROWS: Number of rows (typically 128)
        BLOCK_COLS: Number of columns (typically 4)
    """
    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    r_div_32 = rows // 32
    r_mod_32 = rows % 32

    # Swizzle formula: maps 2D (r, c) to flat index in 512-byte block
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    # Flatten and store
    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS, ))
    scales_flat = tl.reshape(scale_input, (BLOCK_ROWS * BLOCK_COLS, ))

    tl.store(scale_output + output_offset + dest_indices_flat, scales_flat)


# Triton kernel for scale swizzling from torchao
# https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/kernels.py
@triton.jit
def triton_scale_swizzle(
    scale_ptr,
    scale_rows,
    scale_cols,
    output_ptr,
    input_row_stride,
    input_col_stride,
    output_block_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    # Calculate starting row and column for this tile
    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    # Replicate columns instead of zero-padding: wrap column indices to valid range
    # This ensures that when scale_cols < BLOCK_COLS (e.g., HEAD_DIM=64 gives 2 cols),
    # we replicate the valid columns to fill the 4-column block that hardware expects.
    # E.g., for 2 cols: [col0, col1, col0, col1] instead of [col0, col1, 0, 0]
    global_cols = start_col + (cols % scale_cols)

    row_mask = global_rows < scale_rows

    input_scales = tl.load(
        scale_ptr + global_rows * input_row_stride + global_cols * input_col_stride,
        mask=row_mask,
        other=0.0,
    )

    # Calculate block offset for output
    LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
    block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

    # Use shared swizzle function
    swizzle_scales_block(input_scales, output_ptr, block_offset, BLOCK_ROWS, BLOCK_COLS)


def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor:
    """
    Rearranges an E8M0 tensor scale to block-scaled swizzle format.

    This format is suitable for Tmem as described in NVIDIA documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        scale_tensor: Input tensor in row-major format with 8-bit elements [M, K/32]

    Returns:
        Rearranged tensor in block-scaled swizzle format [padded_M, padded_K/32]
    """
    assert scale_tensor.element_size() == 1, "Expected element size to be 1 byte (8 bits)"

    rows, cols = scale_tensor.shape
    BLOCK_ROWS, BLOCK_COLS = 128, 4

    # Calculate blocks needed
    n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    n_col_blocks = triton.cdiv(cols, BLOCK_COLS)
    padded_rows = n_row_blocks * BLOCK_ROWS
    padded_cols = n_col_blocks * BLOCK_COLS

    out = scale_tensor.new_empty((padded_rows, padded_cols))

    # Input stride (for row-major format)
    input_row_stride = scale_tensor.stride()[0]
    input_col_stride = scale_tensor.stride()[1]

    # Output block stride for the rearranged format
    output_block_stride = BLOCK_ROWS * BLOCK_COLS * (padded_cols // BLOCK_COLS)

    grid = lambda META: (
        triton.cdiv(padded_rows, BLOCK_ROWS),
        triton.cdiv(padded_cols, BLOCK_COLS),
    )

    triton_scale_swizzle[grid](
        scale_tensor.view(torch.uint8),
        rows,
        cols,
        out.view(torch.uint8),
        input_row_stride,
        input_col_stride,
        output_block_stride,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )

    return out


# =============================================================================
# Triton in-kernel MXFP8 conversion functions
# =============================================================================


@triton.jit
def _compute_scale_and_quantize(data_block,  # [BLOCK_M, BLOCK_K] float32 input
                                VEC_SIZE: tl.constexpr,  # Block size for scaling (32 for MXFP8)
                                ):
    """
    Compute MXFP8 scales and quantized data for a single block.

    Args:
        data_block: Input tensor of shape [BLOCK_M, BLOCK_K] in float32
        BLOCK_SIZE: The MX block size (typically 32)

    Returns:
        scale_e8m0: E8M0 biased exponent scales [BLOCK_M, BLOCK_K // BLOCK_SIZE]
        data_fp8: Quantized FP8 E4M3 data [BLOCK_M, BLOCK_K]
    """
    # Get dimensions from constexpr
    BLOCK_M: tl.constexpr = data_block.shape[0]
    BLOCK_K: tl.constexpr = data_block.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    # Constants for MXFP8 conversion
    E8M0_EXPONENT_BIAS: tl.constexpr = 127
    FP8_E4M3_MAX: tl.constexpr = 448.0  # torch.finfo(torch.float8_e4m3fn).max

    # Reshape to [BLOCK_M, NUM_SCALES, BLOCK_SIZE] for per-group operations
    data_reshaped = tl.reshape(data_block, [BLOCK_M, NUM_SCALES, VEC_SIZE])

    # Compute max absolute value per group
    # tl.max reduces along the last axis by default
    abs_data = tl.abs(data_reshaped)
    max_abs = tl.max(abs_data, axis=2)  # [BLOCK_M, NUM_SCALES]

    # Compute descale = max_abs / FP8_E4M3_MAX
    descale = max_abs / FP8_E4M3_MAX

    # Compute biased exponent using RCEIL (round up):
    # exponent = clamp(ceil(log2(descale)), -127, 127) + 127
    # Handle special cases: descale == 0 -> exponent = 0
    log2_descale = tl.math.log2(descale)
    ceil_log2 = tl.math.ceil(log2_descale)

    # Clamp to valid E8M0 range [-127, 127] before adding bias
    clamped_exp = tl.maximum(tl.minimum(ceil_log2, E8M0_EXPONENT_BIAS), -E8M0_EXPONENT_BIAS)

    # Handle zero/subnormal case: if descale is 0 or very small, set exponent to 0
    # This handles the case where max_abs is 0
    is_zero = descale < 1e-38
    biased_exp = tl.where(is_zero, 0.0, clamped_exp + E8M0_EXPONENT_BIAS)

    # Convert to uint8 for E8M0 representation
    scale_e8m0 = biased_exp.to(tl.uint8)  # [BLOCK_M, NUM_SCALES]

    # Compute the scaling factor for quantization
    # descale_fp = 2^(127 - biased_exp) = 2^(-unbiased_exp)
    descale_fp = tl.where(biased_exp == 0, 1.0, tl.math.exp2(E8M0_EXPONENT_BIAS - biased_exp))

    # Expand descale_fp for broadcasting: [BLOCK_M, NUM_SCALES, 1]
    descale_fp_expanded = tl.reshape(descale_fp, [BLOCK_M, NUM_SCALES, 1])

    # Scale the data
    scaled_data = data_reshaped * descale_fp_expanded

    # Clamp to FP8 E4M3 representable range
    scaled_data = tl.maximum(tl.minimum(scaled_data, FP8_E4M3_MAX), -FP8_E4M3_MAX)

    # Reshape back to [BLOCK_M, BLOCK_K]
    data_scaled_flat = tl.reshape(scaled_data, [BLOCK_M, BLOCK_K])

    # Cast to FP8 E4M3
    data_fp8 = data_scaled_flat.to(tl.float8e4nv)

    return scale_e8m0, data_fp8


@triton.jit
def to_mxfp8_jit(data,  # [BLOCK_M, BLOCK_K] float32 input tensor
                 BLOCK_SIZE: tl.constexpr = 32,  # MX block size
                 ):
    """
    Convert a float32 tensor to MXFP8 format inside a Triton kernel.

    This function converts float32 data to FP8 E4M3 with E8M0 per-block scales,
    suitable for use with Blackwell's scaled MMA operations.

    Args:
        data: Input tensor of shape [BLOCK_M, BLOCK_K] in float32
        BLOCK_SIZE: The MX block size (typically 32)

    Returns:
        data_fp8: Quantized FP8 E4M3 data [BLOCK_M, BLOCK_K]
        scale_e8m0: E8M0 biased exponent scales [BLOCK_M, BLOCK_K // BLOCK_SIZE]

    Note:
        The scales are returned in row-major format. For TMEM storage with the
        required (1, REP_M, REP_N, 2, 256) layout, use the swizzle_scales_for_tmem
        helper or store with appropriate swizzling.

    Example usage inside a kernel:
        # Load float32 data
        data = tl.load(data_ptr + offsets)

        # Convert to MXFP8
        data_fp8, scales = to_mxfp8(data)

        # Use with scaled_dot...
    """
    # Compute scales and quantized data
    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data, BLOCK_SIZE)

    return data_fp8, scale_e8m0


@triton.jit
def to_mxfp8_with_swizzled_scale(
    data,  # [128, 128] float32 input tensor
    scale_out,  # Preallocated buffer for swizzled scales
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Convert float32 tensor to MXFP8 and produce swizzled scales for TMEM.

    This is the main entry point for converting data inside a Triton kernel
    when the scales need to be stored in TMEM with the (1, REP_M, REP_N, 2, 256) layout.

    Args:
        data: Input tensor of shape [128, 128] in float32
        scale_out: Preallocated SMEM/TMEM buffer for swizzled scale output
        VEC_SIZE: The MX vec size (32)

    Returns:
        data_fp8: Quantized FP8 E4M3 data [128, 128]

    Note:
        Scales are written to scale_out with swizzled layout.
        For 128x128 input with vec=32:
        - NUM_SCALES = 128 / VEC_SIZE = 4
        - REP_M = 128 / 128 = 1
        - REP_N = ceil(4 / 4) = 1
        - Scale output shape: (1, 1, 1, 2, 256)
    """
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE  # 4

    # Step 1: Compute scales and quantized data
    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data, VEC_SIZE)

    # Step 2: Swizzle scales for TMEM using shared function
    swizzle_scales_block(scale_e8m0, scale_out, 0, BLOCK_M, NUM_SCALES)

    return data_fp8


# =============================================================================
# Test and verification code
# =============================================================================


def swizzle_scales_pytorch(scale_tensor: torch.Tensor) -> torch.Tensor:
    """
    Swizzle E8M0 scales to TMEM format using PyTorch.

    Input: [M, K/32] where M=128, K/32=4 for 128x128 input
    Output: [1, REP_M, REP_N, 2, 256] = [1, 1, 1, 2, 256] for 128x128
    """
    rows, cols = scale_tensor.shape
    BLOCK_ROWS, BLOCK_COLS = 128, 4

    n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    n_col_blocks = triton.cdiv(cols, BLOCK_COLS)

    # Pad if needed
    padded_rows = n_row_blocks * BLOCK_ROWS
    padded_cols = n_col_blocks * BLOCK_COLS

    out = scale_tensor.new_zeros((padded_rows, padded_cols))

    # Copy with column replication if cols < BLOCK_COLS
    for r in range(min(rows, padded_rows)):
        for c in range(padded_cols):
            src_c = c % cols  # Replicate columns
            out[r, c] = scale_tensor[r, src_c]

    # Now swizzle
    result = scale_tensor.new_empty((padded_rows * padded_cols, ))

    for r in range(padded_rows):
        for c in range(padded_cols):
            r_div_32 = r // 32
            r_mod_32 = r % 32
            dest_idx = r_mod_32 * 16 + r_div_32 * 4 + c
            result[dest_idx] = out[r, c]

    # Reshape to (1, REP_M, REP_N, 2, 256)
    REP_M = n_row_blocks
    REP_N = n_col_blocks
    result = result.reshape(1, REP_M, REP_N, 2, 256)

    return result


@triton.jit
def _test_to_mxfp8_kernel(
    data_ptr,
    data_fp8_ptr,
    scale_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test kernel that converts data to MXFP8."""
    # Load input data
    row_idx = tl.arange(0, M)[:, None]
    col_idx = tl.arange(0, K)[None, :]
    offsets = row_idx * K + col_idx

    data = tl.load(data_ptr + offsets)

    # Convert to MXFP8
    data_fp8, scale_e8m0 = to_mxfp8_jit(data, BLOCK_SIZE)

    # Store FP8 data
    tl.store(data_fp8_ptr + offsets, data_fp8)

    # Store scales (non-swizzled for now)
    NUM_SCALES: tl.constexpr = K // BLOCK_SIZE
    scale_row_idx = tl.arange(0, M)[:, None]
    scale_col_idx = tl.arange(0, NUM_SCALES)[None, :]
    scale_offsets = scale_row_idx * NUM_SCALES + scale_col_idx

    tl.store(scale_ptr + scale_offsets, scale_e8m0)


@triton.jit
def _test_to_mxfp8_swizzled_kernel(
    data_ptr,
    data_fp8_ptr,
    scale_ptr,  # Output for swizzled scales [1, REP_M, REP_N, 2, 256]
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test kernel that converts data to MXFP8 with swizzled scale output."""
    # Load input data
    row_idx = tl.arange(0, M)[:, None]
    col_idx = tl.arange(0, K)[None, :]
    offsets = row_idx * K + col_idx

    data = tl.load(data_ptr + offsets)

    # Convert to MXFP8 with swizzled scale output
    data_fp8 = to_mxfp8_with_swizzled_scale(data, scale_ptr, BLOCK_SIZE, M, K)

    # Store FP8 data
    tl.store(data_fp8_ptr + offsets, data_fp8)


@triton.jit
def _test_swizzle_scales_kernel(
    scale_in_ptr,
    scale_out_ptr,  # Pre-allocated buffer with shape (1, REP_M, REP_N, 2, 256)
    BLOCK_M: tl.constexpr,
    NUM_SCALES: tl.constexpr,
):
    """Test kernel that applies swizzle_scales_block."""
    # Load input scales [BLOCK_M, NUM_SCALES]
    row_idx = tl.arange(0, BLOCK_M)[:, None]
    col_idx = tl.arange(0, NUM_SCALES)[None, :]
    in_offsets = row_idx * NUM_SCALES + col_idx

    scale_input = tl.load(scale_in_ptr + in_offsets)

    # Call the function being tested - it writes to the output buffer with offset 0
    swizzle_scales_block(scale_input, scale_out_ptr, 0, BLOCK_M, NUM_SCALES)


def test_to_mxfp8_host():
    """Test the PyTorch host-side to_mxfp8 function."""
    torch.manual_seed(42)

    device = torch.device("cuda")

    # Test with different shapes
    test_cases = [
        (128, 128),
        (256, 128),
        (128, 256),
        (64, 64),
    ]

    all_passed = True
    for M, K in test_cases:
        print(f"  Testing shape ({M}, {K})...")

        # Test with float32
        data_f32 = torch.randn(M, K, dtype=torch.float32, device=device)
        scale_f32, data_fp8_f32 = to_mxfp8(data_f32)

        # Verify shapes
        expected_scale_shape = (M, K // 32)
        expected_data_shape = (M, K)

        if scale_f32.shape != expected_scale_shape:
            print(f"    FAIL: scale shape {scale_f32.shape} != expected {expected_scale_shape}")
            all_passed = False
        if data_fp8_f32.shape != expected_data_shape:
            print(f"    FAIL: data shape {data_fp8_f32.shape} != expected {expected_data_shape}")
            all_passed = False

        # Verify dtypes
        if scale_f32.dtype != torch.float8_e8m0fnu:
            print(f"    FAIL: scale dtype {scale_f32.dtype} != torch.float8_e8m0fnu")
            all_passed = False
        if data_fp8_f32.dtype != torch.float8_e4m3fn:
            print(f"    FAIL: data dtype {data_fp8_f32.dtype} != torch.float8_e4m3fn")
            all_passed = False

        # Test with bfloat16
        data_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        scale_bf16, data_fp8_bf16 = to_mxfp8(data_bf16)

        if scale_bf16.shape != expected_scale_shape:
            print(f"    FAIL: bf16 scale shape {scale_bf16.shape} != expected {expected_scale_shape}")
            all_passed = False
        if data_fp8_bf16.shape != expected_data_shape:
            print(f"    FAIL: bf16 data shape {data_fp8_bf16.shape} != expected {expected_data_shape}")
            all_passed = False

        print(f"    Shape ({M}, {K}) passed")

    # Test that quantized values are in valid FP8 range
    data = torch.randn(128, 128, dtype=torch.float32, device=device) * 1000  # Large values
    scale, data_fp8 = to_mxfp8(data)
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    if data_fp8.float().abs().max() > max_fp8:
        print(f"  FAIL: FP8 values exceed max {max_fp8}")
        all_passed = False

    print(f"to_mxfp8 host test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_triton_mx_block_rearrange():
    """Test the triton_mx_block_rearrange function."""
    torch.manual_seed(42)

    device = torch.device("cuda")

    # Test with different shapes
    test_cases = [(128, 4),  # Standard case for 128x128 input
                  (256, 4),  # Larger M
                  (128, 8),  # Larger K (256 input)
                  (256, 8),  # Both larger
                  ]

    all_passed = True
    for rows, cols in test_cases:
        print(f"  Testing scale shape ({rows}, {cols})...")

        # Create random scale data with unique values for easier tracking
        scale_input = torch.randint(0, 255, (rows, cols), dtype=torch.uint8, device=device)

        # Run the swizzle
        scale_output = triton_mx_block_rearrange(scale_input.view(torch.float8_e8m0fnu))

        # Verify output shape
        BLOCK_ROWS, BLOCK_COLS = 128, 4
        n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
        n_col_blocks = triton.cdiv(cols, BLOCK_COLS)
        expected_rows = n_row_blocks * BLOCK_ROWS
        expected_cols = n_col_blocks * BLOCK_COLS

        if scale_output.shape != (expected_rows, expected_cols):
            print(f"    FAIL: output shape {scale_output.shape} != expected ({expected_rows}, {expected_cols})")
            all_passed = False
            continue

        # Verify the swizzle pattern by checking values at known positions
        scale_out_uint8 = scale_output.view(torch.uint8)
        spot_check_passed = True
        mismatches = []

        # Check a subset of positions to verify the swizzle formula
        check_rows = [0, 1, 31, 32, 33, 63, 64, 95, 96, 127]
        check_rows = [r for r in check_rows if r < rows]

        for r in check_rows:
            for c in range(min(cols, BLOCK_COLS)):
                # Compute expected destination using swizzle formula
                r_div_32 = r // 32
                r_mod_32 = r % 32
                dest_idx = r_mod_32 * 16 + r_div_32 * 4 + c

                # The output is flattened row-major in blocks
                # For the first block (rows 0-127, cols 0-3):
                out_row = dest_idx // expected_cols
                out_col = dest_idx % expected_cols

                original_val = scale_input[r, c].item()

                if out_row < expected_rows and out_col < expected_cols:
                    swizzled_val = scale_out_uint8[out_row, out_col].item()

                    if original_val != swizzled_val:
                        mismatches.append((r, c, original_val, swizzled_val, dest_idx))
                        spot_check_passed = False

        if not spot_check_passed:
            print(f"    FAIL: {len(mismatches)} spot check mismatches")
            for r, c, orig, swiz, dest in mismatches[:5]:
                print(f"      input[{r},{c}]={orig} != output[dest={dest}]={swiz}")
            all_passed = False
        else:
            print(f"    Spot checks passed for ({rows}, {cols})")

        # Output may have padding/replication, so just check input values exist
        output_flat = scale_out_uint8.view(-1)
        for val in scale_input.unique():
            val_item = val.item()
            input_count = (scale_input == val_item).sum().item()
            output_count = (output_flat == val_item).sum().item()
            # Output count should be >= input count (may have replication)
            if output_count < input_count:
                print(f"    FAIL: value {val_item} appears {input_count}x in input but only {output_count}x in output")
                all_passed = False

    # Test with actual to_mxfp8 output
    print("  Testing with to_mxfp8 output...")
    data = torch.randn(128, 128, dtype=torch.float32, device=device)
    scale, _ = to_mxfp8(data)
    swizzled = triton_mx_block_rearrange(scale)

    if swizzled.shape != (128, 4):
        print(f"    FAIL: swizzled shape {swizzled.shape} != expected (128, 4)")
        all_passed = False
    else:
        print("    to_mxfp8 integration passed")

    print(f"triton_mx_block_rearrange test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_to_mxfp8_roundtrip():
    """Test that to_mxfp8 + dequantization approximately recovers original values."""
    torch.manual_seed(42)

    device = torch.device("cuda")

    # Create test data with values in a reasonable range
    data = torch.randn(128, 128, dtype=torch.float32, device=device)

    # Quantize
    scale, data_fp8 = to_mxfp8(data)

    # Dequantize manually
    scale_uint8 = scale.view(torch.uint8)
    E8M0_EXPONENT_BIAS = 127

    # Expand scale to match data shape
    scale_expanded = scale_uint8.unsqueeze(-1).expand(-1, -1, 32)
    scale_flat = scale_expanded.reshape(128, 128).float()

    # Compute actual scale values: 2^(exponent - 127)
    scale_values = torch.where(scale_flat == 0, torch.ones_like(scale_flat),
                               torch.exp2(scale_flat - E8M0_EXPONENT_BIAS))

    # Dequantize: fp8_value * scale
    dequantized = data_fp8.float() * scale_values

    # Check relative error (should be within FP8 precision)
    # FP8 E4M3 has ~12.5% relative error for normalized values
    relative_error = (dequantized - data).abs() / (data.abs() + 1e-6)
    max_rel_error = relative_error.max().item()
    mean_rel_error = relative_error.mean().item()

    print(f"  Max relative error: {max_rel_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.4f}")

    # FP8 E4M3 can have up to ~12.5% relative error, plus quantization noise
    # Allow up to 50% for edge cases
    passed = max_rel_error < 0.5 and mean_rel_error < 0.15

    print(f"to_mxfp8 roundtrip test: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_swizzle_scales_block():
    """Test swizzle_scales_block against PyTorch reference.

    Verifies that the swizzled output has the correct (1, REP_M, REP_N, 2, 256) layout
    with proper values at each position.
    """
    torch.manual_seed(42)

    BLOCK_M = 128
    NUM_SCALES = 4  # For 128x128 input with block_size=32

    # Expected output shape dimensions
    REP_M = BLOCK_M // 128  # = 1
    REP_N = (NUM_SCALES + 3) // 4  # = 1
    EXPECTED_SHAPE = (1, REP_M, REP_N, 2, 256)

    device = torch.device("cuda")

    # Create test scale data (random uint8 values simulating E8M0 exponents)
    scale_input = torch.randint(0, 255, (BLOCK_M, NUM_SCALES), dtype=torch.uint8, device=device)

    # Allocate output with the expected 5D shape
    scale_output_5d = torch.zeros(EXPECTED_SHAPE, dtype=torch.uint8, device=device)

    # Run Triton kernel (pass flattened view for pointer arithmetic)
    _test_swizzle_scales_kernel[(1, )](
        scale_input,
        scale_output_5d.view(-1),
        BLOCK_M=BLOCK_M,
        NUM_SCALES=NUM_SCALES,
    )

    # Verify output shape
    assert scale_output_5d.shape == EXPECTED_SHAPE, (
        f"Output shape mismatch: got {scale_output_5d.shape}, expected {EXPECTED_SHAPE}")
    print(f"Output shape correct: {scale_output_5d.shape}")

    # Get PyTorch reference (already in 5D shape)
    scale_ref_5d = swizzle_scales_pytorch(scale_input)
    assert scale_ref_5d.shape == EXPECTED_SHAPE, (
        f"Reference shape mismatch: got {scale_ref_5d.shape}, expected {EXPECTED_SHAPE}")

    # Compare values in 5D layout
    match = torch.equal(scale_output_5d, scale_ref_5d)
    print(f"Swizzle values match: {match}")

    if not match:
        diff_mask = scale_output_5d != scale_ref_5d
        num_mismatches = diff_mask.sum().item()
        print(f"Number of mismatches: {num_mismatches}")

        # Find first few mismatches with their 5D indices
        diff_indices = torch.nonzero(diff_mask)
        print("First 10 mismatch locations (5D indices):")
        for i in range(min(10, len(diff_indices))):
            idx = tuple(diff_indices[i].tolist())
            triton_val = scale_output_5d[idx].item()
            ref_val = scale_ref_5d[idx].item()
            print(f"  {idx}: triton={triton_val}, ref={ref_val}")

    # Additional verification: check that we can recover original values
    # by reverse-swizzling
    print("\nVerifying reverse mapping (spot check):")
    spot_checks_passed = True
    for r in [0, 31, 32, 63, 64, 95, 96, 127]:
        for c in range(NUM_SCALES):
            # Compute expected destination in flat layout
            r_div_32 = r // 32
            r_mod_32 = r % 32
            flat_idx = r_mod_32 * 16 + r_div_32 * 4 + c

            # Map flat index to 5D index
            # flat_idx is within [0, 512), maps to (0, 0, 0, flat_idx//256, flat_idx%256)
            idx_5d = (0, 0, 0, flat_idx // 256, flat_idx % 256)

            original_val = scale_input[r, c].item()
            swizzled_val = scale_output_5d[idx_5d].item()

            if original_val != swizzled_val:
                print(f"  FAIL: input[{r},{c}]={original_val} != output{idx_5d}={swizzled_val}")
                spot_checks_passed = False

    if spot_checks_passed:
        print("  All spot checks passed!")

    return match and spot_checks_passed


def test_to_mxfp8():
    """Test the Triton to_mxfp8 implementation against PyTorch reference."""
    torch.manual_seed(42)

    M, K = 128, 128
    BLOCK_SIZE = 32

    device = torch.device("cuda")

    # Create test data
    data = torch.randn(M, K, dtype=torch.float32, device=device)

    # Allocate outputs
    data_fp8_triton = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device)
    scale_triton = torch.empty(M, K // BLOCK_SIZE, dtype=torch.uint8, device=device)

    # Run Triton kernel
    _test_to_mxfp8_kernel[(1, )](
        data,
        data_fp8_triton,
        scale_triton,
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Get PyTorch reference using to_mxfp8
    scale_ref, data_fp8_ref = to_mxfp8(data, BLOCK_SIZE)
    scale_ref = scale_ref.view(torch.uint8)

    # Compare scales
    scale_match = torch.allclose(scale_triton.float(), scale_ref.float(),
                                 atol=1,  # Allow 1 unit difference due to rounding
                                 )
    print(f"Scale match: {scale_match}")
    if not scale_match:
        diff = (scale_triton.float() - scale_ref.float()).abs()
        print(f"Max scale diff: {diff.max()}")
        print(f"Triton scales sample: {scale_triton[0, :]}")
        print(f"Ref scales sample: {scale_ref[0, :]}")

    # Compare FP8 data
    data_match = torch.allclose(
        data_fp8_triton.float(),
        data_fp8_ref.float(),
        atol=1e-2,
    )
    print(f"Data match: {data_match}")
    if not data_match:
        diff = (data_fp8_triton.float() - data_fp8_ref.float()).abs()
        print(f"Max data diff: {diff.max()}")

    return scale_match and data_match


def test_to_mxfp8_swizzled():
    """Test the Triton to_mxfp8 with swizzled scale output."""
    torch.manual_seed(42)

    M, K = 128, 128
    BLOCK_SIZE = 32
    NUM_SCALES = K // BLOCK_SIZE  # 4
    REP_M = M // 128  # 1
    REP_N = (NUM_SCALES + 3) // 4  # 1

    device = torch.device("cuda")

    # Create test data
    data = torch.randn(M, K, dtype=torch.float32, device=device)

    # Allocate outputs
    data_fp8_triton = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device)
    # Swizzled scale output: (1, REP_M, REP_N, 2, 256) = (1, 1, 1, 2, 256)
    scale_triton = torch.empty(1, REP_M, REP_N, 2, 256, dtype=torch.uint8, device=device)

    # Run Triton kernel
    _test_to_mxfp8_swizzled_kernel[(1, )](
        data,
        data_fp8_triton,
        scale_triton.view(-1),  # Flatten for pointer arithmetic
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Get PyTorch reference using to_mxfp8
    scale_ref, data_fp8_ref = to_mxfp8(data, BLOCK_SIZE)
    scale_ref_uint8 = scale_ref.view(torch.uint8)

    # Swizzle reference scales
    scale_ref_swizzled = swizzle_scales_pytorch(scale_ref_uint8)

    # Compare swizzled scales
    scale_match = torch.equal(scale_triton, scale_ref_swizzled)
    print(f"Swizzled scale match: {scale_match}")
    if not scale_match:
        diff = (scale_triton.float() - scale_ref_swizzled.float()).abs()
        print(f"Max swizzled scale diff: {diff.max()}")
        print(f"Triton swizzled shape: {scale_triton.shape}")
        print(f"Ref swizzled shape: {scale_ref_swizzled.shape}")

    # Compare FP8 data
    data_match = torch.allclose(
        data_fp8_triton.float(),
        data_fp8_ref.float(),
        atol=1e-2,
    )
    print(f"Data match: {data_match}")

    return scale_match and data_match


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch host-side to_mxfp8...")
    print("=" * 60)
    result_host = test_to_mxfp8_host()
    print(f"Test passed: {result_host}\n")

    print("=" * 60)
    print("Testing triton_mx_block_rearrange...")
    print("=" * 60)
    result_rearrange = test_triton_mx_block_rearrange()
    print(f"Test passed: {result_rearrange}\n")

    print("=" * 60)
    print("Testing to_mxfp8 roundtrip (quantize + dequantize)...")
    print("=" * 60)
    result_roundtrip = test_to_mxfp8_roundtrip()
    print(f"Test passed: {result_roundtrip}\n")

    print("=" * 60)
    print("Testing swizzle_scales_block...")
    print("=" * 60)
    result0 = test_swizzle_scales_block()
    print(f"Test passed: {result0}\n")

    print("=" * 60)
    print("Testing to_mxfp8 in-kernel (non-swizzled scales)...")
    print("=" * 60)
    result1 = test_to_mxfp8()
    print(f"Test passed: {result1}\n")

    print("=" * 60)
    print("Testing to_mxfp8 in-kernel with swizzled scales...")
    print("=" * 60)
    result2 = test_to_mxfp8_swizzled()
    print(f"Test passed: {result2}\n")

    print("=" * 60)
    all_passed = all([result_host, result_rearrange, result_roundtrip, result0, result1, result2])
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        print(f"  to_mxfp8_host: {result_host}")
        print(f"  triton_mx_block_rearrange: {result_rearrange}")
        print(f"  to_mxfp8_roundtrip: {result_roundtrip}")
        print(f"  swizzle_scales_block: {result0}")
        print(f"  to_mxfp8 in-kernel: {result1}")
        print(f"  to_mxfp8 in-kernel swizzled: {result2}")
