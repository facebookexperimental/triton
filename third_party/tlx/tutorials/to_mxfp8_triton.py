"""
Triton implementation of to_mxfp8 conversion for use inside Triton kernels.

This module provides a Triton JIT function to convert float32 tensors to MXFP8 format
(FP8 E4M3 data + E8M0 scales), with the output scale swizzled for TMEM usage.

The implementation follows the RCEIL rounding mode from the torchao MXFP8 implementation.
"""

import torch
import triton
import triton.language as tl


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
def _swizzle_scales_for_tmem(
    scale_input,  # [BLOCK_M, NUM_SCALES] uint8 scales
    scale_output,  # Pre-allocated output buffer, flattened (1, REP_M, REP_N, 2, 256)
    BLOCK_M: tl.constexpr,
    NUM_SCALES: tl.constexpr,
):
    """
    Swizzle scales from row-major [BLOCK_M, NUM_SCALES] to TMEM format (1, REP_M, REP_N, 2, 256).

    The TMEM format is (1, REP_M, REP_N, 2, 256) where:
    - REP_M = ceil(BLOCK_M / 128)
    - REP_N = ceil(NUM_SCALES / 4)

    The swizzling follows NVIDIA's block scaling factors layout:
    - 128 rows are grouped into blocks
    - Within each 128-row block, rows are interleaved with columns

    For a 128x4 scale tensor:
    - Input indices: row r (0-127), col c (0-3)
    - Output flat index: (r % 32) * 16 + (r // 32) * 4 + c
    - This flat index maps to 5D as: (0, 0, 0, flat_idx // 256, flat_idx % 256)

    Args:
        scale_input: Input scales tensor [BLOCK_M, NUM_SCALES]
        scale_output: Pre-allocated output buffer pointer (flattened view of (1, REP_M, REP_N, 2, 256))
        BLOCK_M: Number of rows (must be 128 for now)
        NUM_SCALES: Number of scale columns (must be 4 for now)
    """
    # Compute swizzle destination indices
    rows = tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
    cols = tl.arange(0, NUM_SCALES)[None, :]  # [1, NUM_SCALES]

    # Swizzle formula: maps 2D (r, c) to flat index in 512-byte block
    r_div_32 = rows // 32
    r_mod_32 = rows % 32
    dest_idx = r_mod_32 * 16 + r_div_32 * 4 + cols  # [BLOCK_M, NUM_SCALES]

    # Flatten and store
    dest_idx_flat = tl.reshape(dest_idx, [BLOCK_M * NUM_SCALES])
    scale_flat = tl.reshape(scale_input, [BLOCK_M * NUM_SCALES])

    tl.store(scale_output + dest_idx_flat, scale_flat)


@triton.jit
def to_mxfp8(data,  # [BLOCK_M, BLOCK_K] float32 input tensor
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

    # Step 2: Swizzle scales for TMEM
    # scale_e8m0 shape: [128, 4]
    # Need to produce (1, 1, 1, 2, 256) = 512 bytes

    # Swizzle pattern for 128x4 -> 512 flat
    rows = tl.arange(0, BLOCK_M)[:, None]  # [128, 1]
    cols = tl.arange(0, NUM_SCALES)[None, :]  # [1, 4]

    r_div_32 = rows // 32
    r_mod_32 = rows % 32

    # Destination index: r_mod_32 * 16 + r_div_32 * 4 + cols
    dest_idx = r_mod_32 * 16 + r_div_32 * 4 + cols  # [128, 4]
    dest_idx_flat = tl.reshape(dest_idx, [BLOCK_M * NUM_SCALES])
    scale_flat = tl.reshape(scale_e8m0, [BLOCK_M * NUM_SCALES])

    # Store swizzled scales
    tl.store(scale_out + dest_idx_flat, scale_flat)

    return data_fp8


# =============================================================================
# Test and verification code
# =============================================================================


def reference_to_mxfp8_pytorch(data_hp: torch.Tensor, block_size: int = 32):
    """
    Reference PyTorch implementation of to_mxfp8 for verification.
    Extracted from torchao.
    """
    assert data_hp.dtype in (torch.bfloat16, torch.float32)
    assert data_hp.shape[-1] % block_size == 0

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    E8M0_EXPONENT_BIAS = 127

    descale = max_abs / F8E4M3_MAX

    exponent = torch.where(
        torch.isnan(descale),
        0xFF,
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

    data_lp = torch.clamp(data_hp * descale_fp, min=-F8E4M3_MAX, max=F8E4M3_MAX)
    data_lp = data_lp.to(torch.float8_e4m3fn)
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0 = exponent.squeeze(-1)

    return scale_e8m0.view(torch.float8_e8m0fnu), data_lp


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
    data_fp8, scale_e8m0 = to_mxfp8(data, BLOCK_SIZE)

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
    """Test kernel that applies _swizzle_scales_for_tmem."""
    # Load input scales [BLOCK_M, NUM_SCALES]
    row_idx = tl.arange(0, BLOCK_M)[:, None]
    col_idx = tl.arange(0, NUM_SCALES)[None, :]
    in_offsets = row_idx * NUM_SCALES + col_idx

    scale_input = tl.load(scale_in_ptr + in_offsets)

    # Call the function being tested - it writes to the output buffer
    _swizzle_scales_for_tmem(scale_input, scale_out_ptr, BLOCK_M, NUM_SCALES)


def test_swizzle_scales_for_tmem():
    """Test _swizzle_scales_for_tmem against PyTorch reference.

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

    # Get PyTorch reference
    scale_ref, data_fp8_ref = reference_to_mxfp8_pytorch(data, BLOCK_SIZE)
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

    # Get PyTorch reference
    scale_ref, data_fp8_ref = reference_to_mxfp8_pytorch(data, BLOCK_SIZE)
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
    print("Testing _swizzle_scales_for_tmem...")
    result0 = test_swizzle_scales_for_tmem()
    print(f"Test 0 passed: {result0}\n")

    print("Testing to_mxfp8 (non-swizzled scales)...")
    result1 = test_to_mxfp8()
    print(f"Test 1 passed: {result1}\n")

    print("Testing to_mxfp8 with swizzled scales...")
    result2 = test_to_mxfp8_swizzled()
    print(f"Test 2 passed: {result2}\n")

    if result0 and result1 and result2:
        print("All tests passed!")
    else:
        print("Some tests failed!")
