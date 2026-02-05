"""
Helper functions available from either Python or JIT to help simplify working with
MXFP8 data in standard use cases.
"""

import torch

F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
E8M0_EXPONENT_BIAS = 127


# This function is extracted from https://github.com/pytorch/ao/blob/442232fbfb0f6cdfdb9c3eac20f57e5a746ee1bf/torchao/prototype/mx_formats/mx_tensor.py#L94C1-L139C29
def _to_mx_rceil(
    data_hp: torch.Tensor,
    max_abs: torch.Tensor,
    max_pos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A prototype implementation of MXFP scale factor derivation method described in
    https://docs.nvidia.com/cuda/cublas/#d-block-quantization

    For Nvidia GPU with Blackwell+ architecture, the scale factor derivation method
    could be accelerated by the `cvt.rp.satfinite.ue8m0x2.f32` instruction.

    Args:
        data_hp: High precision data.
        max_abs: Maximum absolute value for data_hp along specified dimension/block_size.
        max_pos: The maximum value of the low precision data type.

    Returns:
        exponent: The biased exponent with dtype E8M0 in uint8 container.
        data_lp: The targeted low precision data, in high precision container
            (requires cast to low precision data type).
    """
    descale = max_abs / max_pos
    # TODO: nan/inf needs to be set for any value
    # of nan/inf in input not just amax.
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

    descale_fp = torch.where(exponent == 0, 1.0, torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)))

    # scale and saturated cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
    return exponent, data_lp


# This function is extracted from https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142
def to_mxfp8(data_hp: torch.Tensor, elem_dtype: torch.dtype):
    """
    Scale data_hp a bf16 or fp32 tensor to MXFP8 format using the RCEIL rounding.

    Args:
        data_hp: bf16 or fp32 tensor to be quantized
        elem_dtype: the MXFP8 element dtype to use (either torch.float8_e4m3fn or torch.float8_e5m2)
    Returns:
        Scale: E8M0 biased exponent scales (in float8_e8m0fnu format)
        Data: MXFP8 quantized data (in elem_dtype format)
    """
    assert elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2), f"{elem_dtype} is not supported yet"
    # Originally this was passed as an arg, but we don't want to allow a block size other than 32
    block_size = 32
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

    if elem_dtype == torch.float8_e4m3fn:
        max_pos = F8E4M3_MAX
    else:
        assert elem_dtype == torch.float8_e5m2, f"{elem_dtype} is not supported yet"
        max_pos = F8E5M2_MAX

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    data_lp = data_lp.to(elem_dtype)
    # need to reshape at the end to help inductor fuse things
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp
