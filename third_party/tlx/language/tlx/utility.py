import re
import builtins

import triton.language.core as tl
import triton.runtime.driver as driver
from triton._C.libtriton import amd as _amd


def is_hip():
    target = driver.active.get_current_target()
    return target.backend == "hip"


def is_amd_tdm_target(arch):
    """True iff the AMD backend reports TDM support for ``arch``."""
    if not isinstance(arch, str) or not arch.startswith("gfx"):
        return False
    return _amd.supports_tdm(arch)


def cuda_parse_arch(arch):
    pattern = r"^sm(\d+)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
    return int(match.group(1))


@tl.builtin
def cluster_cta_rank(_semantic=None):
    """
    :return the unique CTA ID within a cluster across all dims
    """
    return tl.tensor(_semantic.builder.create_cluster_cta_rank(), tl.int32)


@tl.builtin
def cluster_size_1d(_semantic=None):
    """
    :return the total number of CTAs in the cluster across all dimensions
    (equal to the product of sizes of every dimension).
    """
    return tl.tensor(_semantic.builder.create_cluster_size_1d(), tl.int32)


@tl.builtin
def thread_id(axis, _semantic=None):
    """
    Returns the id of the current thread instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = tl._unwrap_if_constexpr(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(_semantic.builder.create_thread_id(axis), tl.int32)


@tl.builtin
def num_warps(_semantic=None):
    """Return the number of warps executing the current kernel."""
    return tl.constexpr(_semantic.builder.options.num_warps)


@tl.builtin
def warp_predicate(predicate, inits, body, args=(), wave_uniform=False, _semantic=None, _generator=None):
    """Execute ``body`` under an AMD per-lane execution predicate.

    A scalar predicate supplies the execution bit for the current physical
    lane. For a tensor predicate, elements owned by one lane are OR-reduced to
    that lane's execution bit. A wavefront skips ``body`` when none of its
    lanes are active. Active lanes receive the values returned by
    ``body(*inits, *args)``; inactive lanes keep ``inits``. The body must be an
    ``@triton.jit`` function containing straight-line computation without
    cross-wave synchronization. Bodies containing reductions, dots, or layout
    shuffles must pass ``wave_uniform=True`` to declare that the predicate has
    the same value for every lane in each wavefront. Cross-warp reductions and
    other nested regions are rejected. The body must return exactly the same
    number and types of tensors as ``inits``. A tensor predicate's shape must
    match each carried tensor or a common leading shape (for example, one row
    predicate for row-major matrices).

    This operation has no cross-wave synchronization and is currently lowered
    only by the AMD backend.
    """
    if isinstance(inits, tl.tensor):
        inits = (inits, )
    elif not isinstance(inits, (builtins.tuple, builtins.list, tl.tuple)):
        raise TypeError("warp_predicate inits must be a tensor, tuple, or list")
    if not isinstance(args, (builtins.tuple, builtins.list, tl.tuple)):
        args = (args, )

    inits = builtins.list(inits)
    args = builtins.list(args)
    if not inits:
        raise ValueError("warp_predicate requires at least one carried value")
    if not all(isinstance(value, tl.tensor) for value in inits):
        raise TypeError("warp_predicate carried values must be tensors")

    predicate = _semantic.to_tensor(predicate)
    if predicate.dtype != tl.int1:
        raise TypeError(f"warp_predicate predicate must have bool dtype, got {predicate.dtype}")
    wave_uniform = tl._unwrap_if_constexpr(wave_uniform)
    if not isinstance(wave_uniform, builtins.bool):
        raise TypeError(f"warp_predicate wave_uniform must be a bool, got {type(wave_uniform).__name__}")

    builder = _semantic.builder
    block = builder.new_block()
    insertion_point = builder.get_insertion_point()
    try:
        builder.set_insertion_point_to_start(block)
        body_result = _generator.call_JitFunction(body, inits + args, kwargs={})
        if isinstance(body_result, tl.tensor):
            results = [body_result]
        elif isinstance(body_result, (builtins.tuple, builtins.list, tl.tuple)):
            results = builtins.list(body_result)
        else:
            raise TypeError("warp_predicate body must return a tensor, tuple, or list")
        if len(results) != len(inits):
            raise TypeError(f"warp_predicate body returned {len(results)} values for {len(inits)} carried values")
        for index, (init, result) in enumerate(zip(inits, results)):
            if not isinstance(result, tl.tensor):
                raise TypeError(f"warp_predicate result {index} is not a tensor")
            if result.type != init.type:
                raise TypeError(f"warp_predicate result {index} has type {result.type}, expected {init.type}")
        builder.create_predicate_yield_op([result.handle for result in results])
    finally:
        builder.restore_insertion_point(insertion_point)

    result_types = [result.type.to_ir(builder) for result in results]
    op = builder.create_warp_predicate_op(
        result_types,
        predicate.handle,
        [init.handle for init in inits],
        wave_uniform,
    )
    op.get_region(0).push_back(block)
    builder.set_insertion_point_after(op)
    merged = [tl.tensor(op.get_result(index), result.type) for index, result in enumerate(results)]
    return merged[0] if len(merged) == 1 else builtins.tuple(merged)


@tl.builtin
def async_task_replica_id(_semantic=None):
    from triton.language.extra.tlx.compiler.code_generator import _get_region_replica_id_stack

    region_replica_id_stack = _get_region_replica_id_stack()
    assert len(region_replica_id_stack) > 0, (
        "async_task_replica_id must be called inside an async region where the stack must be non-empty")
    return tl.constexpr(region_replica_id_stack[-1])


@tl.builtin
def dtype_of(v, _semantic=None) -> tl.dtype:
    """
    Returns the element type of a given tensor or tensor descriptor.
    """
    if isinstance(v, tl.tensor):
        dtype = v.type.element_ty
        if dtype.is_ptr():
            dtype = dtype.element_ty
        return dtype
    if isinstance(v, tl.tensor_descriptor_base):
        return v.dtype
    raise ValueError(f"dtype_of only works on tensors and tensor descriptors, but got {v}")


@tl.builtin
def size_of(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    """
    Returns the size of a given dtype.
    """
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(dtype, tl.dtype), f"size_of expects a dtype, but got {type(dtype)}"
    return tl.constexpr(dtype.primitive_bitwidth // 8)


@tl.builtin
def get_fp8_format_name(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    """
    Returns the FP8 format name string for a given FP8 dtype.

    This extracts the format identifier (e.g., "e5m2", "e4m3") from the dtype
    for use with scaled MMA operations like async_dot_scaled.

    Args:
        dtype: An FP8 dtype (tl.float8e5m2 or tl.float8e4nv)

    Returns:
        A constexpr string with the format name ("e5m2" or "e4m3")

    Raises:
        AssertionError: If the dtype is not a supported FP8 type.

    Example:
        Q_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
    """
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(dtype, tl.dtype), f"get_fp8_format_name expects a dtype, but got {type(dtype)}"
    if dtype == tl.float8e5:
        return tl.constexpr("e5m2")
    elif dtype == tl.float8e4nv:
        return tl.constexpr("e4m3")
    else:
        raise AssertionError(f"get_fp8_format_name only supports tl.float8e5 (e5m2) and tl.float8e4nv (e4m3), "
                             f"but got {dtype}")


@tl.builtin
def clock64(_semantic=None):
    """
    Returns the current backend-specific 64-bit hardware timer value.

    Differences between two readings can be used for device-side timing. The
    timer's tick frequency and scope are target-specific. The timer read is not
    a synchronization primitive, so callers timing asynchronous or memory
    operations must issue the appropriate waits or barriers.

    Returns:
        tl.tensor: A tensor containing the current timer value as an int64.

    Example:
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start
    """
    return tl.tensor(_semantic.builder.create_clock64(), tl.int64)


@tl.builtin
def stoch_round(
    src: tl.tensor,
    dst_ty: tl.dtype,
    rand_bits: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Hardware-accelerated stochastic rounding for FP32→FP8/BF16/F16 conversions.

    Requires Blackwell GPU (compute capability >= 100).

    Semantics:
        y = tlx.stoch_round(src, dst_ty, rand_bits)

    Maps to PTX (on Blackwell):
        cvt.rs.satfinite.{e4m3x4,e5m2x4}.f32  d, {a,b,c,d}, rbits  (for FP8)
        cvt.rs.satfinite.{bf16x2,f16x2}.f32   d, {a,b}, rbits      (for BF16/F16)

    Args:
        src:
            Source FP32 tensor. Shape defines output shape.
        dst_ty:
            Destination dtype: tl.float8e5, tl.float8e4nv, tl.float16, or tl.bfloat16
        rand_bits:
            Random bits (uint32 tensor) for entropy, must match src shape

    Returns:
        Tensor with dtype dst_ty and shape matching src.
    """
    capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    assert capability >= 100, (f"stoch_round requires compute capability >= 100 (Blackwell GPU), "
                               f"current capability: {capability}")
    src_ty = src.type
    src_sca_ty = src_ty.scalar

    assert src_sca_ty == tl.float32, (f"Stochastic rounding only supports fp32 source, got {src_sca_ty}. "
                                      f"Source must be float32.")
    assert dst_ty in [tl.float8e5, tl.float8e4nv, tl.float16, tl.bfloat16
                      ], (f"Stochastic rounding only supports fp8/fp16/bf16 destination, got {dst_ty}. "
                          f"Supported types: float8e5 (fp8 E5M2), float8e4nv (fp8 E4M3FN), float16, bfloat16")

    rbits_ty = rand_bits.type
    if src_ty.is_block() and rbits_ty.is_block():
        assert src_ty.shape == rbits_ty.shape, f"rand_bits shape {rbits_ty.shape} must match src shape {src_ty.shape}"
    elif not src_ty.is_block() and not rbits_ty.is_block():
        pass
    else:
        raise ValueError(f"src and rand_bits must both be blocks or both be scalars, "
                         f"got src_ty.is_block()={src_ty.is_block()}, rbits_ty.is_block()={rbits_ty.is_block()}")

    if src_sca_ty == dst_ty:
        return src
    if src_ty.is_block():
        result_ty = src_ty.with_element_ty(dst_ty)
        dst_ir_ty = result_ty.to_ir(_semantic.builder)
    else:
        result_ty = dst_ty
        dst_ir_ty = dst_ty.to_ir(_semantic.builder)
    dst = _semantic.builder.create_cvt_rs(src.handle, dst_ir_ty, rand_bits.handle)
    return tl.tensor(dst, result_ty)
