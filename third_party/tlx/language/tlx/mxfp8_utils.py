"""
Helper functions available from either Python or JIT to help simplify working with
MXFP8 data in standard use cases.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
E8M0_EXPONENT_BIAS = 127


@triton.jit
def _online_swizzle_prefill(
    scale_input,  # [BLOCK_ROWS, BLOCK_COLS] tensor (already loaded)
    scale_output_smem,  # Output Location
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """
    Swizzle a block of scales to 5D format and store to output.

    The swizzling follows NVIDIA's block scaling factors layout:
    - 128 rows are grouped into 4 sub-blocks of 32 rows
    - Swizzle formula: dest_idx = (r % 32) * 16 + (r // 32) * 4 + c

    For a [128, 4] input, the output layout is [32, 16] where:
    - The 128 rows are split into 4 groups of 32 rows
    - Each group of 32 rows is interleaved with the 4 columns

    Args:
        scale_input: Loaded scale tensor [BLOCK_ROWS, BLOCK_COLS]
        scale_output_smem: Output smemm buffer [1, 1, 1, 2, 256]
        BLOCK_ROWS: Number of rows (typically 128)
        BLOCK_COLS: Number of columns (typically 4)
    """
    tl.static_assert(BLOCK_ROWS == 128)
    tl.static_assert(BLOCK_COLS == 4)
    # Optimized SMEM path: use reshape + transpose to swizzle in registers
    # The swizzle formula: dest_idx = (r % 32) * 16 + (r // 32) * 4 + c
    # can be expressed as:
    #   1. Reshape [128, 4] → [4, 32, 4]  (split rows into 4 groups of 32)
    #   2. Transpose to [32, 4, 4]         (interleave the groups)
    #   3. Reshape to output shape
    NUM_GROUPS: tl.constexpr = BLOCK_ROWS // 32  # = 4
    GROUP_SIZE: tl.constexpr = 32

    # Reshape: [BLOCK_ROWS, BLOCK_COLS] → [NUM_GROUPS, GROUP_SIZE, BLOCK_COLS]
    scale_3d = tl.reshape(scale_input, [NUM_GROUPS, GROUP_SIZE, BLOCK_COLS])

    # Transpose: [NUM_GROUPS, GROUP_SIZE, BLOCK_COLS] → [GROUP_SIZE, NUM_GROUPS, BLOCK_COLS]
    scale_transposed = tl.trans(scale_3d, 1, 0, 2)

    # Reshape to 5D output: [1, 1, 1, 2, 256]
    scale_5d = tl.reshape(scale_transposed, [1, 1, 1, 2, 256])
    tlx.local_store(scale_output_smem, scale_5d)


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
    E8M0_EXPONENT_BIAS: tl.constexpr = 127
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
    Convert a float32 tensor to MXFP8 format and store to SMEM.

    This function converts float32 data to FP8 data with E8M0 per-block scales,
    suitable for use with Blackwell's scaled MMA operations. All data stays in
    registers except for the final stores to SMEM.

    Args:
        data_input: Input tensor of shape [BLOCK_M, BLOCK_K] in float32 (in registers)
        data_out_tile: Preallocated SMEM buffer for FP8 data output
        scale_out_tile: Preallocated SMEM buffer for int8 (F8M0) scale output
        VEC_SIZE: The MX block size (typically 32)
        dtype: Target output dtype, either tl.float8e4nv or tl.float8e5

    Note:
        Uses tlx.local_store to write data to SMEM buffers.
        The scale output is swizzled for TMEM format with prefill.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(VEC_SIZE == 32)

    # Step 1: Compute scales and quantized data (all in registers)
    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data_input, VEC_SIZE, dtype)

    # Step 2: Store FP8 data to SMEM
    tlx.local_store(data_out_tile, data_fp8)

    # Step 3: Swizzle and store scales to SMEM
    _online_swizzle_prefill(scale_e8m0, scale_out_tile, BLOCK_M, NUM_SCALES)
