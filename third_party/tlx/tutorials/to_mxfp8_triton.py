"""
Triton implementation of to_mxfp8 conversion for use inside Triton kernels.

This module provides a Triton JIT function to convert float32 tensors to MXFP8 format
(FP8 E4M3 data + E8M0 scales), with the output scale swizzled for TMEM usage.

The implementation follows the RCEIL rounding mode from the torchao MXFP8 implementation.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

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
def swizzle_scales_block(scale_input,  # [BLOCK_ROWS, BLOCK_COLS] tensor (already loaded)
                         scale_output_ptr_or_smem_buffer,  # Output Location
                         output_offset,  # Offset in output buffer for this block
                         BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                         OUTPUT_SMEM: tl.constexpr,  # Whether output is in shared memory
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
        scale_output: Output buffer pointer
        output_offset: Offset in output buffer for this block
        BLOCK_ROWS: Number of rows (typically 128)
        BLOCK_COLS: Number of columns (typically 4)
        OUTPUT_SMEM: Whether output is shared memory (uses tlx.local_store)
    """
    TOTAL_SIZE: tl.constexpr = BLOCK_ROWS * BLOCK_COLS

    if OUTPUT_SMEM:
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
        tlx.local_store(scale_output_ptr_or_smem_buffer, scale_5d)
    else:
        # For global memory output, use scatter-store with computed indices
        rows = tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, BLOCK_COLS)[None, :]

        r_div_32 = rows // 32
        r_mod_32 = rows % 32

        # Swizzle formula: maps 2D (r, c) to flat index in 512-byte block
        dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

        dest_indices_flat = tl.reshape(dest_indices, (TOTAL_SIZE, ))
        scales_flat = tl.reshape(scale_input, (TOTAL_SIZE, ))

        tl.store(scale_output_ptr_or_smem_buffer + output_offset + dest_indices_flat, scales_flat)


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
    swizzle_scales_block(input_scales, output_ptr, block_offset, BLOCK_ROWS, BLOCK_COLS, False)


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
def _to_mxfp8_block(data_input,  # [BLOCK_M, BLOCK_K] float32 input (already in registers)
                    p_tile,  # Preallocated SMEM buffer for FP8 data output
                    p_scale_tile,  # Preallocated SMEM buffer for int8 (F8M0) scale output
                    VEC_SIZE: tl.constexpr,  # MX block size
                    ):
    """
    Convert a float32 tensor to MXFP8 format and store to SMEM.

    This function converts float32 data to FP8 E4M3 with E8M0 per-block scales,
    suitable for use with Blackwell's scaled MMA operations. All data stays in
    registers except for the final stores to SMEM.

    Args:
        data_input: Input tensor of shape [BLOCK_M, BLOCK_K] in float32 (in registers)
        p_tile: Preallocated SMEM buffer for FP8 data output
        p_scale_tile: Preallocated SMEM buffer for int8 (F8M0) scale output
        VEC_SIZE: The MX block size (typically 32)

    Note:
        Uses tlx.local_store to write data to SMEM buffers.
        The scale output is swizzled for TMEM format.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    # Step 1: Compute scales and quantized data (all in registers)
    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data_input, VEC_SIZE)

    # Step 2: Store FP8 data to SMEM
    tlx.local_store(p_tile, data_fp8)

    # Step 3: Swizzle and store scales to SMEM
    swizzle_scales_block(scale_e8m0, p_scale_tile, 0, BLOCK_M, NUM_SCALES, True)
