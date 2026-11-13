import triton.language.core as tl

from .utility import cuda_parse_arch

_SUPPORTED_FP8_TYPES = (tl.float8e4nv, tl.float8e5)
_SUPPORTED_HALF_TYPES = (tl.float16, tl.bfloat16)
_EX2_X2_INSTRUCTIONS = {
    tl.float16: ("ex2.approx.f16x2", 75),
    tl.bfloat16: ("ex2.approx.ftz.bf16x2", 90),
}


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


def _cuda_capability(name, semantic):
    arch = semantic.builder.options.arch
    is_cuda = isinstance(arch, str) and arch.startswith("sm")
    assert is_cuda, f"tlx.{name} requires an NVIDIA GPU, got architecture {arch}"
    return cuda_parse_arch(arch)


def _packed_arith(operation, operands, dtype, semantic):
    capability = _cuda_capability(f"packed_{operation}4", semantic)
    assert capability >= 103, (f"tlx.packed_{operation}4 requires compute capability >= 103, "
                               f"current capability: {capability}")

    operands = tuple(semantic.to_tensor(operand) for operand in operands)
    assert all(operand.type.is_block()
               for operand in operands), (f"tlx.packed_{operation}4 requires block tensor operands")

    for operand in operands:
        assert operand.dtype in _SUPPORTED_FP8_TYPES or _is_packed_fp4(operand), (
            f"tlx.packed_{operation}4 only supports FP8 E4M3/E5M2 or packed FP4 int8 operands, "
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
    assert dtype in _SUPPORTED_FP8_TYPES, (f"tlx.packed_{operation}4 result must be FP8 E4M3 or E5M2, got {dtype}")

    if operation in ("add", "sub"):
        assert operands[1].dtype == dtype, (
            f"tlx.packed_{operation}4 requires the second operand type to match result type {dtype}, "
            f"got {operands[1].dtype}")
    elif operation == "fma":
        assert operands[2].dtype == dtype, (
            f"tlx.packed_fma4 requires the accumulator type to match result type {dtype}, "
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
                f"tlx.packed_{operation}4 packed FP4 operand shape {operand_shape} must equal result shape "
                f"{result_shape} with exactly one dimension halved")
            assert fp4_axis is None or fp4_axis == axis, (
                f"tlx.packed_{operation}4 packed FP4 operands must use the same packed dimension")
            fp4_axis = axis
        else:
            assert operand_shape == result_shape, (
                f"tlx.packed_{operation}4 FP8 operand shape {operand_shape} must match result shape {result_shape}")

    builder = semantic.builder
    handle = builder.create_packed_arith(
        dtype.to_ir(builder),
        operation,
        [operand.handle for operand in operands],
        reference.handle if reference is not None else None,
    )
    return tl.tensor(handle, tl.block_type(dtype, result_shape))


@tl.builtin
def packed_add4(lhs, rhs, dtype=None, _semantic=None):
    """Add four packed FP8/FP4 lanes with an SM103+ PTX 9.4 instruction."""
    return _packed_arith("add", (lhs, rhs), dtype, _semantic)


@tl.builtin
def packed_sub4(lhs, rhs, dtype=None, _semantic=None):
    """Subtract four packed FP8/FP4 lanes with an SM103+ PTX 9.4 instruction."""
    return _packed_arith("sub", (lhs, rhs), dtype, _semantic)


@tl.builtin
def packed_mul4(lhs, rhs, dtype=None, _semantic=None):
    """Multiply four packed FP8/FP4 lanes with an SM103+ PTX 9.4 instruction.

    When both operands are packed FP4, ``dtype`` is required and packing is
    inferred along the final tensor dimension.
    """
    return _packed_arith("mul", (lhs, rhs), dtype, _semantic)


@tl.builtin
def packed_fma4(lhs, rhs, acc, dtype=None, _semantic=None):
    """Fused multiply-add four packed FP8/FP4 lanes with an SM103+ PTX 9.4 instruction."""
    return _packed_arith("fma", (lhs, rhs, acc), dtype, _semantic)


@tl.builtin
def packed_exp2(x, _semantic=None):
    """Approximate 2**x on FP16/BF16 tensors, two lanes per ``ex2.approx`` instruction.

    Rubin (SM107) doubles FP16 exp throughput relative to FP32.
    """
    x = _semantic.to_tensor(x)
    assert x.dtype in _EX2_X2_INSTRUCTIONS, f"tlx.packed_exp2 only supports FP16/BF16, got {x.dtype}"
    instruction, min_capability = _EX2_X2_INSTRUCTIONS[x.dtype]
    capability = _cuda_capability("packed_exp2", _semantic)
    assert capability >= min_capability, (f"tlx.packed_exp2 on {x.dtype} requires compute capability >= "
                                          f"{min_capability}, current capability: {capability}")
    return tl.inline_asm_elementwise(
        f"{instruction} $0, $1;",
        "=r,r",
        [x],
        dtype=x.dtype,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@tl.builtin
def packed_add2(lhs, rhs, _semantic=None):
    """Add an FP16/BF16 tensor to an FP32 tensor, producing FP32.

    On SM107 this lowers to the mixed-precision ``add.f32x2.{f16x2,bf16x2}.f32x2``
    (PTX 9.4) and, like ``packed_add4``, needs operands with a distributed layout
    (e.g. from a TMEM load or ``tlx.require_layout``); earlier targets upcast and
    use an ordinary FP32 add.
    """
    lhs = _semantic.to_tensor(lhs)
    rhs = _semantic.to_tensor(rhs)
    if lhs.dtype == tl.float32:
        lhs, rhs = rhs, lhs
    assert lhs.dtype in _SUPPORTED_HALF_TYPES and rhs.dtype == tl.float32, (
        f"tlx.packed_add2 requires one FP16/BF16 and one FP32 operand, got {lhs.dtype} and {rhs.dtype}")
    assert lhs.type.is_block() and rhs.type.is_block(), "tlx.packed_add2 requires block tensor operands"
    assert lhs.type.shape == rhs.type.shape, (
        f"tlx.packed_add2 operand shapes must match, got {lhs.type.shape} and {rhs.type.shape}")

    if _cuda_capability("packed_add2", _semantic) != 107:
        return _semantic.add(_semantic.cast(lhs, tl.float32), rhs, True)
    builder = _semantic.builder
    handle = builder.create_packed_arith(tl.float32.to_ir(builder), "add", [lhs.handle, rhs.handle], rhs.handle)
    return tl.tensor(handle, rhs.type)
