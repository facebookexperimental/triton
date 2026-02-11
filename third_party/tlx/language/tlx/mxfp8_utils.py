"""
Helper functions available from either Python or JIT to help simplify working with
MXFP8 data in standard use cases.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _compute_scale_and_quantize(
    data_block,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Compute MXFP8 scales and quantized data for a single block.

    Args:
        data_block: Input tensor of shape [BLOCK_M, BLOCK_K] in float32
        BLOCK_SIZE: The MX block size (typically 32)
        dtype: Target output dtype, either tl.float or torch.float8_e5m2

    Returns:
        scale_e8m0: E8M0 biased exponent scales [BLOCK_M, BLOCK_K // BLOCK_SIZE]
        data_fp8: Quantized FP8 E4M3 data [BLOCK_M, BLOCK_K]
    """
    # Get dimensions from constexpr
    BLOCK_M: tl.constexpr = data_block.shape[0]
    BLOCK_K: tl.constexpr = data_block.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    # Constants for MXFP8 conversion
    if dtype == tl.float8e4nv:
        # torch.finfo(torch.float8_e4m3fn).max
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        # torch.finfo(torch.float8_e5m2).max
        FLOAT_MAX: tl.constexpr = 57344.0

    # Reshape to [BLOCK_M, NUM_SCALES, BLOCK_SIZE] for per-group operations
    data_reshaped = tl.reshape(data_block, [BLOCK_M, NUM_SCALES, VEC_SIZE])

    # Compute max absolute value per group
    # tl.max reduces along the last axis by default
    abs_data = tl.abs(data_reshaped)
    max_abs = tl.max(abs_data, axis=2)  # [BLOCK_M, NUM_SCALES]

    # Compute descale = max_abs / FLOAT_MAX
    descale = max_abs / FLOAT_MAX

    # Round descale up to the next power of 2 using exact bit manipulation (RCEIL).
    # Adding 0x007FFFFF bumps the exponent by 1 unless the mantissa is already zero
    # (i.e., the value is already an exact power of 2). This avoids precision issues
    # with the floating-point log2/ceil approach.
    descale_exponent = (descale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    descale_rounded = descale_exponent.to(tl.float32, bitcast=True)

    # Extract E8M0 biased exponent: the IEEE 754 exponent field >> 23
    scale_e8m0 = (descale_exponent >> 23).to(tl.uint8)  # [BLOCK_M, NUM_SCALES]

    # Compute the quantization scale (reciprocal of the dequant scale).
    # When descale_rounded is 0 (all values in the block are zero), use 0 to zero out data.
    quant_scale = tl.where(descale_rounded == 0, 0.0, 1.0 / descale_rounded)

    # Expand quant_scale for broadcasting: [BLOCK_M, NUM_SCALES, 1]
    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])

    # Scale the data
    scaled_data = data_reshaped * quant_scale_expanded

    # Clamp to FP8 E4M3 representable range
    scaled_data = tl.maximum(tl.minimum(scaled_data, FLOAT_MAX), -FLOAT_MAX)

    # Reshape back to [BLOCK_M, BLOCK_K]
    data_scaled_flat = tl.reshape(scaled_data, [BLOCK_M, BLOCK_K])

    # Cast to FP8 E4M3
    data_fp8 = data_scaled_flat.to(dtype)

    return scale_e8m0, data_fp8


@triton.jit
def _to_mxfp8_block(
    data_input,
    data_out_tile,
    scale_out_tile,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Convert a float32 tensor to MXFP8 format and store results.

    This function converts float32 data to FP8 data with E8M0 per-block scales,
    suitable for use with Blackwell's scaled MMA operations. All data stays in
    registers except for the final stores.

    Args:
        data_input: Input tensor of shape [BLOCK_M, BLOCK_K] in float32 (in registers)
        data_out_tile: Preallocated SMEM buffer for FP8 data output
        scale_out_tile: Preallocated buffer for int8 (E8M0) scale output (SMEM or TMEM)
        VEC_SIZE: The MX block size (typically 32)
        dtype: Target output dtype, either tl.float8e4nv or tl.float8e5

    Note:
        Uses tlx.local_store to write data and scales to their respective buffers.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(VEC_SIZE == 32)

    # Step 1: Compute scales and quantized data (all in registers)
    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data_input, VEC_SIZE, dtype)

    # Step 2: Store FP8 data to SMEM
    tlx.local_store(data_out_tile, data_fp8)

    # Step 3: Store scales
    tlx.local_store(scale_out_tile, scale_e8m0)
