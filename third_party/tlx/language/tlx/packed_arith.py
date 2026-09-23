import triton.language.core as tl

from .utility import cuda_parse_arch

_SUPPORTED_FP8_TYPES = (tl.float8e4nv, tl.float8e5)


def _is_packed_fp4(operand):
    return operand.dtype.is_int() and operand.dtype.primitive_bitwidth == 8


def _packed_fp4_axis(operand_shape, result_shape):
    if len(operand_shape) != len(result_shape):
        return None
    packed_axes = [
        axis for axis, (operand_extent, result_extent) in enumerate(zip(operand_shape, result_shape))
        if result_extent == operand_extent * 2
    ]
    if len(packed_axes) != 1:
        return None
    packed_axis = packed_axes[0]
    if any(operand_extent != result_extent
           for axis, (operand_extent, result_extent) in enumerate(zip(operand_shape, result_shape))
           if axis != packed_axis):
        return None
    return packed_axis


def _packed_arith(operation, operands, dtype, semantic):
    arch = semantic.builder.options.arch
    assert isinstance(arch, str) and arch.startswith("sm"), (
        f"tlx.{operation}4 requires an NVIDIA Rubin GPU, got architecture {arch}")
    capability = cuda_parse_arch(arch)
    assert capability >= 107, (f"tlx.{operation}4 requires compute capability >= 107 (Rubin GPU), "
                               f"current capability: {capability}")

    operands = tuple(semantic.to_tensor(operand) for operand in operands)
    assert all(operand.type.is_block() for operand in operands), (f"tlx.{operation}4 requires block tensor operands")

    for operand in operands:
        assert operand.dtype in _SUPPORTED_FP8_TYPES or _is_packed_fp4(operand), (
            f"tlx.{operation}4 only supports FP8 E4M3/E5M2 or packed FP4 int8 operands, "
            f"got {operand.dtype}")

    dtype = tl._unwrap_if_constexpr(dtype)
    fp8_operands = [operand for operand in operands if operand.dtype in _SUPPORTED_FP8_TYPES]
    addend = operands[-1]
    if operation in ("add", "sub", "fma") and addend.dtype in _SUPPORTED_FP8_TYPES:
        reference = addend
    else:
        reference = fp8_operands[0] if fp8_operands else None
    if dtype is None:
        if operation in ("add", "sub", "fma") and addend.dtype in _SUPPORTED_FP8_TYPES:
            dtype = addend.dtype
        elif reference is not None:
            dtype = reference.dtype
        else:
            raise ValueError("packed FP4 operands require an explicit FP8 result dtype")

    assert isinstance(dtype, tl.dtype), f"expected 'dtype' to be a dtype but got {dtype}"
    assert dtype in _SUPPORTED_FP8_TYPES, (f"tlx.{operation}4 result must be FP8 E4M3 or E5M2, got {dtype}")

    if operation in ("add", "sub"):
        assert operands[1].dtype == dtype, (
            f"tlx.{operation}4 requires the second operand type to match result type {dtype}, "
            f"got {operands[1].dtype}")
    elif operation == "fma":
        assert operands[2].dtype == dtype, (f"tlx.fma4 requires the accumulator type to match result type {dtype}, "
                                            f"got {operands[2].dtype}")

    if reference is None:
        result_shape = list(operands[0].type.shape)
        result_shape[-1] *= 2
    else:
        result_shape = list(reference.type.shape)

    fp4_axis = None
    for operand in operands:
        operand_shape = list(operand.type.shape)
        if _is_packed_fp4(operand):
            axis = _packed_fp4_axis(operand_shape, result_shape)
            assert axis is not None, (
                f"tlx.{operation}4 packed FP4 operand shape {operand_shape} must equal result shape "
                f"{result_shape} with exactly one dimension halved")
            assert fp4_axis is None or fp4_axis == axis, (
                f"tlx.{operation}4 packed FP4 operands must use the same packed dimension")
            fp4_axis = axis
        else:
            assert operand_shape == result_shape, (
                f"tlx.{operation}4 FP8 operand shape {operand_shape} must match result shape {result_shape}")

    builder = semantic.builder
    handle = builder.create_packed_arith(
        dtype.to_ir(builder),
        operation,
        [operand.handle for operand in operands],
        reference.handle if reference is not None else None,
    )
    return tl.tensor(handle, tl.block_type(dtype, result_shape))


@tl.builtin
def add4(lhs, rhs, dtype=None, _semantic=None):
    """Add four packed FP8/FP4 lanes with a Rubin instruction."""
    return _packed_arith("add", (lhs, rhs), dtype, _semantic)


@tl.builtin
def sub4(lhs, rhs, dtype=None, _semantic=None):
    """Subtract four packed FP8/FP4 lanes with a Rubin instruction."""
    return _packed_arith("sub", (lhs, rhs), dtype, _semantic)


@tl.builtin
def mul4(lhs, rhs, dtype=None, _semantic=None):
    """Multiply four packed FP8/FP4 lanes with a Rubin instruction.

    When both operands are packed FP4, ``dtype`` is required and packing is
    inferred along the final tensor dimension.
    """
    return _packed_arith("mul", (lhs, rhs), dtype, _semantic)


@tl.builtin
def fma4(lhs, rhs, acc, dtype=None, _semantic=None):
    """Fused multiply-add four packed FP8/FP4 lanes with a Rubin instruction."""
    return _packed_arith("fma", (lhs, rhs, acc), dtype, _semantic)
