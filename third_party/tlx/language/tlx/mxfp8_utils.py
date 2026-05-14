"""
Helper functions available from either Python or JIT to help simplify working with
MXFP8 data in standard use cases.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fused_amax_to_e8m0(amax, max_norm_rcp):
    """
    Fused amax-to-E8M0 scale conversion in a single PTX asm block.

    Computes E8M0 biased exponent (RCEIL of amax / max_norm) and the
    reciprocal quantization scale (power-of-two inv_scale) in one pass,
    replacing ~8 separate Python/Triton operations.

    Returns (e8m0_exp as uint32, inv_scale as float32).
    Caller should cast e8m0_exp to uint8.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 fae_scale;
            .reg .u32 fae_bits, fae_exp, fae_mantissa, fae_inv_exp, fae_inv_bits;
            .reg .pred fae_has_mantissa;
            mul.f32 fae_scale, $2, $3;
            mov.b32 fae_bits, fae_scale;
            bfe.u32 fae_exp, fae_bits, 23, 8;
            and.b32 fae_mantissa, fae_bits, 0x7FFFFF;
            setp.ne.u32 fae_has_mantissa, fae_mantissa, 0;
            @fae_has_mantissa add.u32 fae_exp, fae_exp, 1;
            min.u32 fae_exp, fae_exp, 254;
            mov.u32 $0, fae_exp;
            sub.u32 fae_inv_exp, 254, fae_exp;
            shl.b32 fae_inv_bits, fae_inv_exp, 23;
            mov.b32 $1, fae_inv_bits;
        }
        """,
        "=r,=f,f,f",
        [amax, max_norm_rcp],
        dtype=(tl.uint32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e4m3x4_f32(a):
    """
    Vectorized FP32 → FP8 E4M3 conversion using packed cvt.rn.satfinite.e4m3x2
    instructions. Converts 4 float32 values to 4 packed FP8 values, avoiding
    scalar conversions and PRMT byte-permute instructions.

    The satfinite modifier saturates to ±448 (e4m3 max), eliminating the need
    for an explicit clamp.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
            cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
            mov.b32 $0, {lo, hi};
        }
        """,
        "=r,f,f,f,f",
        [a],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _cvt_e5m2x4_f32(a):
    """Vectorized FP32 → FP8 E5M2 conversion. See _cvt_e4m3x4_f32."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e5m2x2.f32 lo, $2, $1;
            cvt.rn.satfinite.e5m2x2.f32 hi, $4, $3;
            mov.b32 $0, {lo, hi};
        }
        """,
        "=r,f,f,f,f",
        [a],
        dtype=tl.float8e5,
        is_pure=True,
        pack=4,
    )


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
        VEC_SIZE: The MX block size (typically 32)
        dtype: Target output dtype, either tl.float8e4nv or tl.float8e5

    Returns:
        scale_e8m0: E8M0 biased exponent scales [BLOCK_M, BLOCK_K // VEC_SIZE]
        data_fp8: Quantized FP8 data [BLOCK_M, BLOCK_K]
    """
    BLOCK_M: tl.constexpr = data_block.shape[0]
    BLOCK_K: tl.constexpr = data_block.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    if dtype == tl.float8e4nv:
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        FLOAT_MAX: tl.constexpr = 57344.0

    data_reshaped = tl.reshape(data_block, [BLOCK_M, NUM_SCALES, VEC_SIZE])

    abs_data = tl.abs(data_reshaped)
    max_abs = tl.max(abs_data, axis=2)  # [BLOCK_M, NUM_SCALES]

    scale_u32, quant_scale = _fused_amax_to_e8m0(max_abs, 1.0 / FLOAT_MAX)
    scale_e8m0 = scale_u32.to(tl.uint8)

    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])
    scaled_data = data_reshaped * quant_scale_expanded
    data_scaled_flat = tl.reshape(scaled_data, [BLOCK_M, BLOCK_K])

    if dtype == tl.float8e4nv:
        data_fp8 = _cvt_e4m3x4_f32(data_scaled_flat)
    else:
        data_fp8 = _cvt_e5m2x4_f32(data_scaled_flat)

    return scale_e8m0, data_fp8


@triton.jit
def _to_mxfp8_block(
    data_input,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Convert a float32 tensor to MXFP8 format.

    This function converts float32 data to FP8 data with E8M0 per-block scales,
    suitable for use with Blackwell's scaled MMA operations.

    Args:
        data_input: Input tensor of shape [BLOCK_M, BLOCK_K] in float32 (in registers)
        VEC_SIZE: The MX block size (typically 32)
        dtype: Target output dtype, either tl.float8e4nv or tl.float8e5

    Return:
        The FP8 data and E8M0 scales. Callers are responsible for storing them.
    """
    BLOCK_K: tl.constexpr = data_input.shape[1]
    tl.static_assert(VEC_SIZE == 32)
    tl.static_assert(BLOCK_K % VEC_SIZE == 0)

    scale_e8m0, data_fp8 = _compute_scale_and_quantize(data_input, VEC_SIZE, dtype)

    return data_fp8, scale_e8m0


@triton.jit
def _amax_to_e8m0_and_quantize(
    data_input,
    block_amax,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Compute E8M0 scales from pre-computed block amaxes and quantize data to FP8.

    Instead of computing max(abs(data)) per block (128 max ops per row), this
    function accepts pre-computed block amaxes derived from the raw QK values
    via monotonicity of exp2: max(exp2(x)) == exp2(max(x)).

    Args:
        data_input: Input tensor [BLOCK_M, BLOCK_K] in float32
        block_amax: Pre-computed block amaxes [BLOCK_M, NUM_SCALES]
        VEC_SIZE: MX block size (32)
        dtype: tl.float8e4nv or tl.float8e5

    Returns:
        scale_e8m0: E8M0 biased exponent scales [BLOCK_M, NUM_SCALES]
        data_fp8: Quantized FP8 data [BLOCK_M, BLOCK_K]
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE

    if dtype == tl.float8e4nv:
        FLOAT_MAX: tl.constexpr = 448.0
    else:
        tl.static_assert(dtype == tl.float8e5)
        FLOAT_MAX: tl.constexpr = 57344.0

    scale_u32, quant_scale = _fused_amax_to_e8m0(block_amax, 1.0 / FLOAT_MAX)
    scale_e8m0 = scale_u32.to(tl.uint8)

    data_reshaped = tl.reshape(data_input, [BLOCK_M, NUM_SCALES, VEC_SIZE])
    quant_scale_expanded = tl.reshape(quant_scale, [BLOCK_M, NUM_SCALES, 1])
    scaled_data = data_reshaped * quant_scale_expanded
    data_scaled_flat = tl.reshape(scaled_data, [BLOCK_M, BLOCK_K])

    if dtype == tl.float8e4nv:
        data_fp8 = _cvt_e4m3x4_f32(data_scaled_flat)
    else:
        data_fp8 = _cvt_e5m2x4_f32(data_scaled_flat)

    return scale_e8m0, data_fp8


@triton.jit
def _to_mxfp8_block_with_block_amax(
    data_input,
    block_amax,
    VEC_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Convert float32 data to MXFP8 using pre-computed block amaxes.

    This is the blockscaled variant of _to_mxfp8_block that skips the expensive
    max(abs(data)) computation per 32-element block by accepting pre-computed
    block amaxes derived from raw QK values.

    Args:
        data_input: Input tensor [BLOCK_M, BLOCK_K] in float32
        block_amax: Pre-computed block amaxes [BLOCK_M, NUM_SCALES]
        VEC_SIZE: MX block size (32)
        dtype: tl.float8e4nv or tl.float8e5

    Return:
        The FP8 data and E8M0 scales. Callers are responsible for storing them.
    """
    BLOCK_M: tl.constexpr = data_input.shape[0]
    BLOCK_K: tl.constexpr = data_input.shape[1]
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(VEC_SIZE == 32)

    scale_e8m0, data_fp8 = _amax_to_e8m0_and_quantize(data_input, block_amax, VEC_SIZE, dtype)

    return data_fp8, scale_e8m0
